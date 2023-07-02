import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.search.operations import *
from src.search.genotypes import NA_PRIMITIVES_v0, NA_PRIMITIVES_v1, NA_PRIMITIVES_v2, NA_PRIMITIVES_v3, SC_PRIMITIVES, \
    LA_PRIMITIVES
import logging


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


def geno_map(version):
    if version == "v0":
        NA_PRIMITIVES = NA_PRIMITIVES_v0
        return NA_PRIMITIVES
    elif version == "v1":
        NA_PRIMITIVES = NA_PRIMITIVES_v1
        return NA_PRIMITIVES
    elif version == "v2":
        NA_PRIMITIVES = NA_PRIMITIVES_v2
        return NA_PRIMITIVES
    elif version == "v3":
        NA_PRIMITIVES = NA_PRIMITIVES_v3
        return NA_PRIMITIVES
    else:
        raise Exception("wrong NA_PRIMITIVES version")


class NaMixedOp(nn.Module):

    def __init__(self, in_dim, out_dim, with_linear, NA_PRIMITIVES):
        super(NaMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.with_linear = with_linear
        self.in_dim = in_dim

        for primitive in NA_PRIMITIVES:
            op = NA_OPS[primitive](in_dim, out_dim)
            self._ops.append(op)

            if with_linear:
                self._ops_linear = nn.ModuleList()
                op_linear = torch.nn.Linear(in_dim, out_dim)
                self._ops_linear.append(op_linear)

    def forward(self, x, x0, weights, edge_index, ):
        mixed_res = []
        if self.with_linear:
            for w, op, linear in zip(weights, self._ops, self._ops_linear):
                mixed_res.append(w * F.elu(op(x, x0, edge_index) + linear(x)))
        else:
            for w, op in zip(weights, self._ops):
                mixed_res.append(w * F.elu(op(x, x0, edge_index)))
        return sum(mixed_res)


class ScMixedOp(nn.Module):

    def __init__(self):
        super(ScMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in SC_PRIMITIVES:
            op = SC_OPS[primitive]()
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(x))
        return sum(mixed_res)


class LaMixedOp(nn.Module):

    def __init__(self, hidden_size, num_layers=None):
        super(LaMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in LA_PRIMITIVES:
            op = LA_OPS[primitive](hidden_size, num_layers)
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * F.elu(op(x)))
        return sum(mixed_res)


class Network(nn.Module):
    '''
        implementation of Auto-HeG
    '''

    def __init__(self, criterion, in_dim, out_dim, hidden_size, device, num_layers=3, dropout=0.5,
                 switches_na=[], with_conv_linear=False, args=None):
        super(Network, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._criterion = criterion
        self.dropout = dropout
        self.with_linear = with_conv_linear
        self.device = device
        self.args = args
        self.NA_num_edges = 3
        if args.fix_last:
            self.SC_num_edges=2
        else:
            self.SC_num_edges = 3
        self.LA_num_edges = 1
        self.num_edges = self.NA_num_edges + self.SC_num_edges + self.LA_num_edges
        self.NA_PRIMITIVES = geno_map(self.args.space_ver)

        self.tau = args.tau_max

        self.switches_na = switches_na
        switch_ons = []
        for i in range(len(switches_na)):
            ons = 0
            for j in range(len(switches_na[i])):
                if switches_na[i][j]:
                    ons = ons + 1
            switch_ons.append(ons)
            ons = 0

        self.switch_on = switch_ons[0]

        # node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)
        self.layer1 = NaMixedOp(hidden_size, hidden_size, self.with_linear, self.NA_PRIMITIVES)
        self.layer2 = NaMixedOp(hidden_size, hidden_size, self.with_linear, self.NA_PRIMITIVES)
        self.layer3 = NaMixedOp(hidden_size, hidden_size, self.with_linear, self.NA_PRIMITIVES)

        # skip op
        self.layer4 = ScMixedOp()
        self.layer5 = ScMixedOp()
        if not self.args.fix_last:
            self.layer6 = ScMixedOp()

        # layer aggregator op
        self.layer7 = LaMixedOp(hidden_size, num_layers)

        self.classifier = nn.Linear(hidden_size, out_dim)

        self._initialize_alphas()

        self._initialize_flags()

        self._initialize_proj_weights()

    def new(self):
        model_new = Network(self._criterion, self.in_dim, self.out_dim, self.hidden_size).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def _initialize_flags(self):
        if self.args.fix_last:
            self.candidate_flags = {
                'NA_PRIM': torch.tensor(3 * [True], requires_grad=False, dtype=torch.bool).to(self.device),
                'SC_PRIM': torch.tensor(2 * [True], requires_grad=False, dtype=torch.bool).to(self.device),
                'LA_PRIM': torch.tensor(1 * [True], requires_grad=False, dtype=torch.bool).to(self.device),
            }  # must be in this order

        else:
            self.candidate_flags = {
                'NA_PRIM': torch.tensor(3 * [True], requires_grad=False, dtype=torch.bool).to(self.device),
                'SC_PRIM': torch.tensor(3 * [True], requires_grad=False, dtype=torch.bool).to(self.device),
                'LA_PRIM': torch.tensor(1 * [True], requires_grad=False, dtype=torch.bool).to(self.device)
            }  # must be in this order

    def _initialize_proj_weights(self):
        ''' data structures used for proj '''
        if isinstance(self.na_alphas, list):
            alphas_na = torch.stack(self.na_alphas, dim=0)
            alphas_sc = torch.stack(self.sc_alphas, dim=0)
            alphas_la = torch.stack(self.la_alphas, dim=0)
        else:
            alphas_na = self.na_alphas
            alphas_sc = self.sc_alphas
            alphas_la = self.la_alphas

        self.proj_weights = {  # for hard/soft assignment after project
            'NA_PRIM': torch.zeros_like(alphas_na).to(self.device),
            'SC_PRIM': torch.zeros_like(alphas_sc).to(self.device),
            'LA_PRIM': torch.zeros_like(alphas_la).to(self.device),
        }

    ## proj function
    def project_op(self, eid, opid, cell_type):
        self.proj_weights[cell_type][eid][opid] = 1  ## hard by default
        self.candidate_flags[cell_type][eid] = False

    ## critical function
    def get_projected_weights(self, cell_type):
        ''' used in forward and genotype '''
        weights = self.get_softmax()[cell_type]
        if cell_type == 'NA_PRIM':
            ## proj op
            for eid in range(3):
                if not self.candidate_flags[cell_type][eid]:
                    weights[eid].data.copy_(self.proj_weights[cell_type][eid])

        if cell_type == 'SC_PRIM':
            if self.args.fix_last:
                for eid in range(2):
                    if not self.candidate_flags[cell_type][eid]:
                        weights[eid].data.copy_(self.proj_weights[cell_type][eid])
            else:
                for eid in range(3):
                    if not self.candidate_flags[cell_type][eid]:
                        weights[eid].data.copy_(self.proj_weights[cell_type][eid])

        elif cell_type == 'LA_PRIM':
            ## proj op
            for eid in range(1):
                if not self.candidate_flags[cell_type][eid]:
                    weights[eid].data.copy_(self.proj_weights[cell_type][eid])

        return weights

    def forward(self, data, weights_dict=None):

        if self.args.edge_index == 'mixhop':
            x, edge_index = data.x, data.mix_edge_index
        elif self.args.edge_index == 'treecomp':
            x, edge_index = data.x, data.tree_edge_index
        elif self.args.edge_index == 'origin':
            edge_index = []
            x = data.x
            edge_index.append(data.edge_index)
            edge_index.append(data.edge_index)
            edge_index.append(data.edge_index)

        if weights_dict is None or 'NA_PRIM' not in weights_dict:
            weights_NA = self.na_alphas
        else:
            weights_NA = weights_dict['NA_PRIM']
        if weights_dict is None or 'SC_PRIM' not in weights_dict:
            weights_SC = self.sc_alphas
        else:
            weights_SC = weights_dict['SC_PRIM']

        if weights_dict is None or 'LA_PRIM' not in weights_dict:
            weights_LA = self.la_alphas
        else:
            weights_LA = weights_dict['LA_PRIM']

        def get_gumbel_prob(xins):
            while True:
                gumbels = -torch.empty_like(xins).exponential_().log()
                logits = (xins.log_softmax(dim=1) + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (
                        (torch.isinf(gumbels).any())
                        or (torch.isinf(probs).any())
                        or (torch.isnan(probs).any())
                ):
                    continue
                else:
                    break
            return probs, hardwts, index



        na_alphas_probs, na_alphas_hardwts, na_alphas_index = get_gumbel_prob(weights_NA)
        sc_alphas_probs, sc_alphas_hardwts, sc_alphas_index = get_gumbel_prob(weights_SC)
        la_alphas_probs, la_alphas_hardwts, la_alphas_index = get_gumbel_prob(weights_LA)

        na_weights = na_alphas_probs
        sc_weights = sc_alphas_probs
        la_weights = la_alphas_probs


        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x0 = x
        x1 = self.layer1(x, x0, na_weights[0], edge_index[0])
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.layer2(x1, x0, na_weights[1], edge_index[1])
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.layer3(x2, x0, na_weights[2], edge_index[2])
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        if self.args.fix_last:
            x4 = (x3, self.layer4(x1, sc_weights[0]), self.layer5(x2, sc_weights[1]))
        else:
            x4 = (self.layer4(x1, sc_weights[0]), self.layer5(x2, sc_weights[1]),
                  self.layer6(x3, sc_weights[2]))

        x5 = self.layer7(x4, la_weights[0])
        x5 = F.dropout(x5, p=self.dropout, training=self.training)

        logits = self.classifier(x5)
        return logits

    def _loss(self, data, is_valid=True):
        logits = self(data)
        if is_valid:
            input = logits[data.val_mask].to(self.device)
            target = data.y[data.val_mask].to(self.device)
        else:
            input = logits[data.train_mask].to(self.device)
            target = data.y[data.train_mask].to(self.device)
        return self._criterion(input, target)

    def _initialize_alphas(self):
        num_na_ops = self.switch_on
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)
        self.na_alphas = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(3, num_na_ops)))
        if self.args.fix_last:
            self.sc_alphas = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(2, num_sc_ops)))
        else:
            self.sc_alphas = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(3, num_sc_ops)))
        self.la_alphas = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(1, num_la_ops)))
        self._arch_parameters = [
            self.na_alphas,
            self.sc_alphas,
            self.la_alphas,
        ]

    def arch_parameters(self):
        return self._arch_parameters
        
    def set_arch_parameters(self, new_alphas):
        for alpha, new_alpha in zip(self.arch_parameters(), new_alphas):
            alpha.data.copy_(new_alpha.data)

    def genotype(self):

        def _parse(na_weights, sc_weights, la_weights, NA_PRIMITIVES):
            gene = []
            na_indices = torch.argmax(na_weights, dim=-1)
            for k in na_indices:
                gene.append(NA_PRIMITIVES[k])
            sc_indices = torch.argmax(sc_weights, dim=-1)
            for k in sc_indices:
                gene.append(SC_PRIMITIVES[k])
            la_indices = torch.argmax(la_weights, dim=-1)
            for k in la_indices:
                gene.append(LA_PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.na_alphas, dim=-1).data.cpu(), F.softmax(self.sc_alphas, dim=-1).data.cpu(),
                      F.softmax(self.la_alphas, dim=-1).data.cpu(), self.NA_PRIMITIVES)

        return gene

    def genotype_pt(self):
        def _parse(na_weights, sc_weights, la_weights, NA_PRIMITIVES):
            gene = []
            na_indices = torch.argmax(na_weights, dim=-1)
            for k in na_indices:
                gene.append(NA_PRIMITIVES[k])
            sc_indices = torch.argmax(sc_weights, dim=-1)
            for k in sc_indices:
                gene.append(SC_PRIMITIVES[k])
            la_indices = torch.argmax(la_weights, dim=-1)
            for k in la_indices:
                gene.append(LA_PRIMITIVES[k])
            return '||'.join(gene)

        weights_NA = self.get_projected_weights('NA_PRIM')
        weights_SC = self.get_projected_weights('SC_PRIM')
        weights_LA = self.get_projected_weights('LA_PRIM')
        gene_pt = _parse(weights_NA.data.cpu(), weights_SC.data.cpu(),
                      weights_LA.data.cpu(), self.NA_PRIMITIVES)

        return gene_pt

    def get_softmax(self):
        weights_NA = F.softmax(self.na_alphas, dim=-1)
        weights_SC = F.softmax(self.sc_alphas, dim=-1)
        weights_LA = F.softmax(self.la_alphas, dim=-1)
        return {'NA_PRIM': weights_NA, 'SC_PRIM': weights_SC, 'LA_PRIM': weights_LA}

    def get_dict(self):
        return {'NA_PRIM': self.na_alphas, 'SC_PRIM': self.sc_alphas, 'LA_PRIM': self.la_alphas}

    def get_state_dict(self, epoch):
        model_state_dict = {
            'epoch': epoch,  ## no +1 because we are saving before projection / at the beginning of an epoch
            'state_dict': self.state_dict(),
            'alpha': self.arch_parameters(),
            'candidate_flags': self.candidate_flags,
            'proj_weights': self.proj_weights,
        }
        return model_state_dict

    def set_state_dict(self, checkpoint):
        ## common
        self.load_state_dict(checkpoint['state_dict'])
        self.set_arch_parameters(checkpoint['alpha'])
        self.candidate_flags = checkpoint['candidate_flags']
        self.proj_weights = checkpoint['proj_weights']

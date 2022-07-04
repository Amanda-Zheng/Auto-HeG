import torch
import torch.nn as nn
from src.search.operations import *


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

class NaOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act, with_linear=False):
    super(NaOp, self).__init__()

    self._op = NA_OPS[primitive](in_dim, out_dim)
    self.primitive = primitive
    self.op_linear = nn.Linear(in_dim, out_dim)
    self.act = act_map(act)
    self.with_linear = with_linear

  def forward(self, x, x0, edge_index):

    if self.with_linear:
        return self.act(self._op(x, x0, edge_index)+self.op_linear(x))
    else:
        return self.act(self._op(x, x0, edge_index))


class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)

class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size, act, num_layers=None):
        super(LaOp, self).__init__()
        self._op = LA_OPS[primitive](hidden_size, num_layers)
        self.act = act_map(act)

    def forward(self, x):
        return self.act(self._op(x))

class NetworkGNN(nn.Module):
    '''
        implementation of AutoHeG
    '''
    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', is_mlp=False, args=None):
        super(NetworkGNN, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self._criterion = criterion
        ops = genotype.split('||')
        self.args = args

        #node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)

        self.gnn_layers = nn.ModuleList(
                [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear) for i in range(num_layers)])

        #skip op

        if self.args.fix_last:
            if self.num_layers > 1:
                self.sc_layers = nn.ModuleList([ScOp(ops[i+num_layers]) for i in range(num_layers - 1)])
            else:
                self.sc_layers = nn.ModuleList([ScOp(ops[num_layers])])
        else:
            # no output conditions.
            skip_op = ops[num_layers:2 * num_layers]
            if skip_op == ['none'] * num_layers:
                skip_op[-1] = 'skip'
            self.sc_layers = nn.ModuleList([ScOp(skip_op[i]) for i in range(num_layers)])


        #layer aggregator op
        self.layer6 = LaOp(ops[-1], hidden_size, 'linear', num_layers)

        self.classifier = nn.Linear(hidden_size, out_dim)

    def forward(self, data):
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

        #generate weights by softmax
        x = self.lin1(x)
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        x0 = x
        js = []
        for i in range(self.num_layers):
            x = self.gnn_layers[i](x, x0,edge_index[i])
            if self.args.with_layernorm:
                layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
                x = layer_norm(x)
            x = F.dropout(x, p=self.in_dropout, training=self.training)
            if i == self.num_layers - 1 and self.args.fix_last:
                js.append(x)
            else:
                js.append(self.sc_layers[i](x))
        x5 = self.layer6(js)
        x5 = F.dropout(x5, p=self.out_dropout, training=self.training)

        logits = self.classifier(x5)
        return logits

    def _loss(self, logits, target):
        return self._criterion(logits, target)


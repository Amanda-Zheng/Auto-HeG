import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge
from torch_geometric.nn import GINConv
from torch_geometric.nn import ChebConv, APPNP, GCN2Conv, FAConv, SGConv, SuperGATConv
from src.model_lib.pyg_gnn_layer import GeoLayer
from src.model_lib.geniepath import GeniePathLayer
from src.model_lib.gprgnn import GPR_prop

NA_OPS = {
    'gcnii': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcnii'),
    'cheb': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cheb'),
    'appnp': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'appnp'),
    'gprgnn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gprgnn'),
    'sgc': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sgc'),
    'supergat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'supergat'),
    'fagcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'fagcn'),
    'sage': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sage'),
    'sage_sum': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sum'),
    'sage_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'max'),
    'gcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn'),
    'gat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat'),
    'gin': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin'),
    'gat_sym': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat_sym'),
    'gat_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'linear'),
    'gat_cos': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cos'),
    'gat_generalized_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'generalized_linear'),
    'geniepath': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'geniepath'),
}

SC_OPS = {
    'none': lambda: Zero(),
    'skip': lambda: Identity(),
}

LA_OPS = {
    'l_max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
    'l_concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
    'l_lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers)
}


class NaAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator):
        super(NaAggregator, self).__init__()
        self.in_dim = in_dim
        self.aggregator = aggregator
        if 'gcnii' == self.aggregator:
            self._op = GCN2Conv(in_dim, alpha=0.5, add_self_loops=False, normalize=True)
            # note that this add self-loop part can be omit.
        if 'cheb' == self.aggregator:
            self._op = ChebConv(in_dim, out_dim, K=5)
            # note that this k need to be well set. denote the filter size, which is the recursively computed.
        if 'appnp' == self.aggregator:
            self._op = APPNP(K=2, alpha=0.5, add_self_loops=False, normalize=True)
            # note that this k is also iterative number.
        if 'gprgnn' == self.aggregator:
            self._op = GPR_prop(K=2, alpha=1.0, Init='PPR', Gamma=None)
            # note that this k is also iterative number.
        if 'sgc' == self.aggregator:
            self._op = SGConv(in_dim, out_dim, K=1, add_self_loops=False)
            # here k is the number of hops
        if 'supergat' == self.aggregator:
            heads = 8
            out_dim /= heads
            self._op = SuperGATConv(in_dim, int(out_dim), add_self_loops=False, heads=heads, dropout=0.5)
        if 'fagcn' == self.aggregator:
            self._op = FAConv(channels=in_dim, add_self_loops=False)

        if 'sage' == self.aggregator:
            self._op = SAGEConv(in_dim, out_dim, normalize=True)
        if 'gcn' == self.aggregator:
            self._op = GCNConv(in_dim, out_dim, add_self_loops=False)
        if 'gat' == self.aggregator:
            heads = 8
            out_dim /= heads
            self._op = GATConv(in_dim, int(out_dim), add_self_loops=False, heads=heads, dropout=0.5)
        if 'gin' == self.aggregator:
            nn1 = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
            self._op = GINConv(nn1)
        if self.aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
            heads = 8
            out_dim /= heads
            self._op = GeoLayer(in_dim, int(out_dim), heads=heads, att_type=aggregator, dropout=0.5)
        if self.aggregator in ['sum', 'max']:
            self._op = GeoLayer(in_dim, out_dim, att_type='const', agg_type=aggregator, dropout=0.5)
        if self.aggregator in ['geniepath']:
            self._op = GeniePathLayer(in_dim, out_dim)

    def forward(self, x, x0, edge_index):
        if self.aggregator == 'fagcn' or self.aggregator == 'gcnii':
            return self._op(x, x0, edge_index)
        else:
            return self._op(x, edge_index)


class LaAggregator(nn.Module):

    def __init__(self, mode, hidden_size, num_layers=3):
        super(LaAggregator, self).__init__()
        self.jump = JumpingKnowledge(mode, channels=hidden_size, num_layers=num_layers)
        if mode == 'cat':
            self.lin = Linear(hidden_size * num_layers, hidden_size)
        else:
            self.lin = Linear(hidden_size, hidden_size)

    def forward(self, xs):
        return self.lin(F.relu(self.jump(xs)))


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)

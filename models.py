import torch
from layers import GraphConvolution
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import math

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class gfNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(gfNN, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nout)
        self.dropout = dropout

    def forward(self, x):
        # x = torch.sparse.mm(adj, x)
        # x = torch.sparse.mm(adj, x)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        return x

class rotator(nn.Module):
    def __init__(self, nfeat):
        super(rotator, self).__init__()
        self.linear = nn.Linear(nfeat, nfeat)

    def forward(self, x):
        x = self.linear(x)
        return x

class simple(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super().__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj=None):
        x = F.relu(self.linear1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        return x

import torch
from layers import GraphConvolution, GraphAttentionLayer
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
import math

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GAT, self).__init__()
        self.gc1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=0.2)
        self.gc2 = GraphAttentionLayer(nhid, nout, dropout=dropout, alpha=0.2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

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

class littleGCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(littleGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
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


class MLP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super().__init__()
        self.linear1 = torch.nn.Linear(nfeat, nhid)
        self.linear2 = torch.nn.Linear(nhid, nhid)
        self.linear3 = torch.nn.Linear(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.linear1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear3(x)
        return x

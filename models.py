import torch
from layers import GraphConvolution
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import math

# # classic
# class GCN(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nout, dropout):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(nfeat, nhid, add_self_loops=False)
#         self.conv2 = GCNConv(nhid, nout, add_self_loops=False)
#         self.dropout = dropout
#
#     def forward(self, data, batch=None):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = self.conv1(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout)
#         x = self.conv2(x, edge_index, edge_weight=edge_weight)
#         return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

        # self.gc2 = GraphConvolution(nhid1, nhid2)
        # self.gc3 = GraphConvolution(nhid2, nhid2)
        # self.gc4 = GraphConvolution(nhid2, nhid2)

        self.gc5 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc4(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc5(x, adj)
        return x

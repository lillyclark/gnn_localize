from process_dataset import *
from models import *
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric

num_nodes = 4
num_anchors = 3
threshold = 3

true_locs = torch.Tensor([[0,0],[0,2],[2,0],[2,2]])
distance_matrix = matrix_from_locs(true_locs)
noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
noise.fill_diagonal_(0)
noisy_distance_matrix = distance_matrix + noise
adj = (noisy_distance_matrix<2.1).float()
adj = (adj + adj.T)/2
adj = normalize(adj, p=1.0, dim=1)

node_ids = torch.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# node_ids = torch.Tensor([[0],[1],[2],[3]])

# model = simple(nfeat=num_nodes, nhid=4, nout=2, dropout=0.0)
model = GCN(nfeat=num_nodes, nhid=4, nout=2, dropout=0.0)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
loss_fn = torch.nn.MSELoss()

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    pred = model(node_ids, adj)
    pred_matrix = matrix_from_locs(pred)
    loss_train = loss_fn(pred_matrix, noisy_distance_matrix) + loss_fn(pred[:num_anchors], true_locs[:num_anchors])
    loss_train.backward()
    optimizer.step()
    if epoch%100==0:
        print("loss_train:", loss_train.item())
print("final loss_train:", loss_train.item())
print("final loss_val:", loss_fn(pred[num_anchors:], true_locs[num_anchors:]).item())

model.eval()
pred = model(node_ids, adj)

print("Ouput:",pred.detach().numpy())

plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
plt.scatter(true_locs[:,0].detach().numpy(), true_locs[:,1].detach().numpy(), label="actual")
plt.legend()
plt.title('four nodes demo')
plt.show()

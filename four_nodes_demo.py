from process_dataset import *
from models import *
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric

np.random.seed(1)
torch.manual_seed(1)

num_nodes = 4
num_anchors = 3
threshold = 10

true_locs = torch.Tensor([[0,0],[0,2],[2,0],[2,2]])
distance_matrix = matrix_from_locs(true_locs)
noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
noise.fill_diagonal_(0)
noisy_distance_matrix = distance_matrix + noise

### threshold adj
adj = (noisy_distance_matrix<threshold).float()
# adj.fill_diagonal_(0)
adj = (adj + adj.T)/2
print(adj)
adj = normalize(adj+torch.eye(num_nodes), p=1.0, dim=1)

### reciprocal distance adj
# adj = 1/noisy_distance_matrix
# adj.fill_diagonal_(0)
# adj = (adj + adj.T)/2
# adj = normalize(adj, p=1.0, dim=1)
# adj.fill_diagonal_(1)

### feature matrix
features = noisy_distance_matrix.clone()
features[features>threshold] = 0
features = (features + features.T)/2
features = normalize(features, p=1.0, dim=1)

node_ids = torch.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# node_ids = torch.Tensor([[0],[1],[2],[3]])

model = simple(nfeat=num_nodes, nhid=8, nout=2, dropout=0.0)
# model = GCN(nfeat=num_nodes, nhid=8, nout=2, dropout=0.0)
# model = littleGCN(nfeat=num_nodes, nhid=num_nodes, nout=2, dropout=0.0)
# model = GAT(nfeat=num_nodes, nhid=8, nout=2, dropout=0.0)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
loss_fn = torch.nn.MSELoss()

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    # pred = model(node_ids, adj)
    pred = model(features, adj)

    pred_matrix = matrix_from_locs(pred)
    # loss_train = loss_fn(pred_matrix, noisy_distance_matrix) + loss_fn(pred[:num_anchors], true_locs[:num_anchors])
    loss_train = loss_fn(pred[:num_anchors], true_locs[:num_anchors])
    loss_train.backward()
    optimizer.step()
    if epoch%100==0:
        print("loss_train:", loss_train.item())
    if loss_train.item() < 0.0005:
        print("stopping after", epoch, "epochs")
        break
print("final loss_train:", loss_train.item())
print("final loss_val:", loss_fn(pred[num_anchors:], true_locs[num_anchors:]).item())

model.eval()
# pred = model(node_ids, adj)
pred = model(features, adj)

print("Ouput:",pred.detach().numpy())

c = ["red","orange","green","blue"]
for n in [0,1,2,3]:
    plt.scatter(pred[n,0].detach().numpy(), pred[n,1].detach().numpy(), label=str(n), color=c[n])
    plt.scatter(true_locs[n,0].detach().numpy(), true_locs[n,1].detach().numpy(), label=str(n)+"true", color=c[n], marker="x")
plt.legend()
plt.title('four nodes demo')
plt.show()

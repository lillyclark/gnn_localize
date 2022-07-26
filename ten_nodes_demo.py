import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric

from models import littleGCN, GCN
from process_dataset import normalize, matrix_from_locs
from lle_like import neighbors, barycenter_weights, weight_to_mat

np.random.seed(1)
torch.manual_seed(1)

num_nodes = 10
num_anchors = 5
n_neighbors = 3
threshold = 3

true_locs = torch.Tensor([[0,0],[0,2],[2,0],[2,2],[1,1],[0,1],[1,0],[1,2],[2,1],[-1,-1]])
distance_matrix = matrix_from_locs(true_locs)
noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
noise.fill_diagonal_(0)
noisy_distance_matrix = distance_matrix + noise

# model = GCN(nfeat=num_nodes, nhid=20, nout=2, dropout=0.0)
model = littleGCN(nfeat=num_nodes, nhid=None, nout=2, dropout=None)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
loss_fn = torch.nn.MSELoss()

mode = 'novel'

if mode=='baseline':
    print("using baseline")
    features = noisy_distance_matrix.clone()
    features[features>threshold] = 0
    features = torch.Tensor(normalize(features,use_sparse=False))
    adj = (noisy_distance_matrix<threshold).float()
    adj.fill_diagonal_(0)
    print(adj)
    adj = torch.Tensor(normalize(adj+torch.eye(num_nodes),use_sparse=False))
    print(np.round(adj.numpy(),2))

elif mode=='novel':
    print("using barycenter adj")
    print("using one hot encoding")
    features = torch.eye(num_nodes)
    # features = torch.Tensor(normalize(noisy_distance_matrix,use_sparse=False))
    indices = neighbors(noisy_distance_matrix, n_neighbors)
    weights = barycenter_weights(noisy_distance_matrix, indices, reg=1e-3)
    adj = torch.Tensor(weight_to_mat(weights,indices))
    print(np.round(adj.numpy(),2))

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    pred = model(features, adj)

    pred_matrix = matrix_from_locs(pred)
    # loss_train = loss_fn(pred_matrix, noisy_distance_matrix) + loss_fn(pred[:num_anchors], true_locs[:num_anchors])
    loss_train = loss_fn(pred[:num_anchors], true_locs[:num_anchors])
    loss_train.backward()
    optimizer.step()
    if epoch%10==0:
        print("loss_train:", loss_train.item())
    if loss_train.item() < 0.0005:
        print("stopping after", epoch, "epochs")
        break
print("final loss_train:", loss_train.item())
print("final loss_val:", loss_fn(pred[num_anchors:], true_locs[num_anchors:]).item())

model.eval()
pred = model(features, adj)

print("pred:")
print(pred)

c = ["red","orange","yellow","green","blue","pink","purple","cyan","magenta","grey"]
for n in range(num_nodes):
    plt.scatter(pred[n,0].detach().numpy(), pred[n,1].detach().numpy(), label=str(n), color=c[n])
    plt.scatter(true_locs[n,0].detach().numpy(), true_locs[n,1].detach().numpy(), label=str(n)+" true", color=c[n], marker="x")
plt.legend()
plt.title('ten nodes demo')
plt.show()

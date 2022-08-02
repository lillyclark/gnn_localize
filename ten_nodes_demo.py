import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric

from models import littleGCN, GCN
from process_dataset import normalize, matrix_from_locs
from lle_like import solve_like_LLE, neighbors, barycenter_weights, weight_to_mat
from decomposition import normalize_tensor, reduce_rank, denoise_via_SVD

np.random.seed(1)
torch.manual_seed(1)

num_nodes = 10
num_anchors = 5
n_neighbors = 5
threshold = 1

true_locs = torch.Tensor([[0,0],[0,2],[2,0],[2,2],[1,1],[0,1],[1,0],[1,2],[2,1],[-1,-1]])
distance_matrix = matrix_from_locs(true_locs)
noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
noise.fill_diagonal_(0)
noisy_distance_matrix = distance_matrix + noise

model = GCN(nfeat=num_nodes, nhid=20, nout=2, dropout=0.0)
# model = littleGCN(nfeat=num_nodes, nhid=None, nout=2, dropout=None)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
loss_fn = torch.nn.MSELoss()

features = noisy_distance_matrix.clone()
features[features>threshold] = 0
features = torch.Tensor(normalize(features,use_sparse=False))
adj = (noisy_distance_matrix<threshold).float()
adj = normalize_tensor(adj)
print("normalized augmented adjacency matrix")
print(np.round(adj.numpy(),2))

# GCN #########################

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    pred = model(features, adj)

    # pred_matrix = matrix_from_locs(pred)
    # loss_train = loss_fn(pred_matrix, noisy_distance_matrix) + loss_fn(pred[:num_anchors], true_locs[:num_anchors])
    loss_train = loss_fn(pred[:num_anchors], true_locs[:num_anchors])
    loss_train.backward()
    optimizer.step()
    if epoch%20==0:
        print("loss_train:", loss_train.item())
    if loss_train.item() < 0.0005:
        print("stopping after", epoch, "epochs")
        break
print("final loss_train:", loss_train.item())

model.eval()
pred = model(features, adj)
print("final loss_val:", loss_fn(pred[num_anchors:], true_locs[num_anchors:]).item())
err = torch.sqrt(loss_fn(pred[num_anchors:], true_locs[num_anchors:])).item()
print(f"test (RMSE):{err}")

print("pred:")
print(pred)

fig, ax = plt.subplots(1,2)
c = ["red","orange","yellow","green","blue","pink","purple","cyan","magenta","grey"]

for n in range(num_nodes):
    ax[0].scatter(pred[n,0].detach().numpy(), pred[n,1].detach().numpy(), label=str(n), color=c[n])
    ax[0].scatter(true_locs[n,0].detach().numpy(), true_locs[n,1].detach().numpy(), label=str(n)+" true", color=c[n], marker="x")
# ax[0].legend()
ax[0].set_title(f'GCN RMSE: {np.round(err,3)}')

# NOVEL #################################

euclidean_matrix = noisy_distance_matrix**2
noisy_distance_matrix = denoise_via_SVD(euclidean_matrix,k=4,fill_diag=False,take_sqrt=False)
anchor_locs = true_locs[:num_anchors]
pred = solve_like_LLE(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=True)
pred = torch.Tensor(pred)
print("final loss_val:", loss_fn(pred[num_anchors:], true_locs[num_anchors:]).item())
err = torch.sqrt(loss_fn(pred[num_anchors:], true_locs[num_anchors:])).item()
print(f"test (RMSE):{err}")

for n in range(num_nodes):
    ax[1].scatter(pred[n,0].detach().numpy(), pred[n,1].detach().numpy(), label=str(n), color=c[n])
    ax[1].scatter(true_locs[n,0].detach().numpy(), true_locs[n,1].detach().numpy(), label=str(n)+" true", color=c[n], marker="x")
# ax[1].legend()
ax[1].set_title(f'novel RMSE: {np.round(err,2)}')

print("pred:")
print(pred)

##########################################

plt.legend()
plt.suptitle(f'{num_nodes} Nodes ({num_anchors} Anchors), Threshold {threshold}, {n_neighbors} Neighbors')
plt.show()

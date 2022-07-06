from process_dataset import *
from models import *
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

np.random.seed(0)
torch.manual_seed(0)

num_nodes = 100
num_anchors = 90
threshold = 1
data_loader, num_nodes = no_noise_dataset(num_nodes, num_anchors, threshold=threshold)

loss_fn = torch.nn.MSELoss()

for batch in data_loader:
    x = batch.x.to_dense().numpy()
    x = (x+x.T)/2
    print(np.round(x,2))

    M = np.zeros(x.shape)

    for i in range(num_nodes):
        for j in range(num_nodes):
            M[i][j] = (x[0][j]**2 + x[i][0]**2 - x[i][j]**2)/2

    q, v = np.linalg.eig(M)
    locs = np.zeros((num_nodes,2))
    locs[:,0] = np.sqrt(q[0])*v[:,0]
    locs[:,1] = np.sqrt(q[1])*v[:,1]

    pca = PCA(n_components=2)
    anchor_mean = torch.mean(batch.y[batch.anchors], axis=0)
    print(anchor_mean)
    anchor_pca = torch.Tensor(pca.fit_transform(batch.y[batch.anchors]))
    anchor_M = pca.components_

    locs = pca.fit_transform(locs)
    locs = np.matmul(locs, anchor_M)
    locs = torch.Tensor(locs + anchor_mean.numpy())


    pred = torch.Tensor(locs)
    loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
    print(f"train:{loss_train.item()}, val:{loss_val.item()}")

plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual")
plt.legend()
plt.title('decomposition')
plt.show()

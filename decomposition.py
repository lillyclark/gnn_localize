from process_dataset import process_dataset, fake_dataset
from models import GCN, gfNN
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

# data_loader, num_nodes = process_dataset('datasets/comp1_clean.csv', batch_size=1, threshold=1000, fake_links=False)
# num_batches = len(data_loader)
# print(num_batches, "batches")

num_nodes = 10
num_anchors = 5
threshold = 1
data_loader, num_nodes = fake_dataset(num_nodes, num_anchors, threshold=threshold)

loss_fn = torch.nn.MSELoss()

for batch in data_loader:
    x = batch.x.to_dense().numpy()
    x = (x+x.T)/2

    M = np.zeros(x.shape)

    for i in range(num_nodes):
        for j in range(num_nodes):
            M[i][j] = (x[0][j]**2 + x[i][0]**2 - x[i][j]**2)/2

    print(np.round(M,2))
    print("rank of M:", np.linalg.matrix_rank(M))

    q, v = np.linalg.eig(M)
    locs = np.zeros((num_nodes,2))
    locs[:,0] = np.sqrt(q[0])*v[:,0]
    locs[:,1] = np.sqrt(q[1])*v[:,1]
    print(np.round(locs,2))

    pca = PCA(n_components=2)
    batch.y = torch.Tensor(pca.fit_transform(batch.y))
    locs = torch.Tensor(pca.fit_transform(locs))

    pred = torch.Tensor(locs)
    loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
    print(f"train:{loss_train.item()}, val:{loss_val.item()}")

plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual")
plt.legend()
plt.title('scoped')
plt.show()

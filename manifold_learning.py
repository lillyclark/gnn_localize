from process_dataset_new import process_dataset, fake_dataset
from models import GCN
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

num_nodes = 50
num_anchors = 10
threshold = 6

# data_loader, num_nodes = process_dataset('datasets/comp1_clean.csv', batch_size=1, threshold=1000, fake_links=False)
# num_batches = len(data_loader)
# print(num_batches, "batches")
start = time.time()
data_loader, num_nodes = fake_dataset(num_nodes, num_anchors, threshold=threshold)
print(f"loaded data in {time.time()-start} secs")

loss_fn = torch.nn.MSELoss()

start = time.time()
for batch in data_loader:
    print(batch)

    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=np.random.RandomState(seed=0), dissimilarity="precomputed", n_jobs=1)
    x = batch.x.to_dense().numpy()
    print(np.round(x,2))
    print("making x symmetric")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if j > i:
                x[j][i]  = x[i][j]
    pred = mds.fit(x).embedding_
    # print(batch.x)
    # print(batch.adj)
    # print(batch.y)
    # print(batch.anchors)

    print(pred)

    pred *= np.sqrt((batch.y.numpy()**2).sum()) / np.sqrt((pred**2).sum())
    clf = PCA(n_components=2)
    batch.y = torch.Tensor(clf.fit_transform(batch.y.numpy()))
    pred = clf.fit_transform(pred)

    pred = torch.Tensor(pred)
    loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
    print(f"train:{loss_train.item()}, val:{loss_val.item()}")

loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual")
plt.legend()
plt.title('Manifold')
plt.show()

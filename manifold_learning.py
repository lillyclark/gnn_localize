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

num_nodes = 50
num_anchors = 10
threshold = 6

def scale_to(input, tensor_to_match, mask):
    for dim in range(input.shape[1]):
        input[:,dim] = input[:,dim] * (torch.max(tensor_to_match[mask][:,dim]) - torch.min(tensor_to_match[mask][:,dim]))/(torch.max(input[mask][:,dim]) - torch.min(input[mask][:,dim]))
        input[:,dim] = input[:,dim] + torch.mean(tensor_to_match[mask][:,dim])
    return input

# data_loader, num_nodes = process_dataset('datasets/comp1_clean.csv', batch_size=1, threshold=1000, fake_links=False)
# num_batches = len(data_loader)
# print(num_batches, "batches")
start = time.time()
data_loader, num_nodes = fake_dataset(num_nodes, num_anchors, threshold=threshold)
print(f"loaded data in {time.time()-start} secs")

model = gfNN(nfeat=2, nhid=2, nout=2, dropout=0)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
loss_fn = torch.nn.MSELoss()

start = time.time()
for batch in data_loader:
    x = batch.x.to_dense().numpy()
    x = (x + x.T)/2
    mds = manifold.MDS(n_components=2, metric=True, max_iter=3000, eps=1e-9, random_state=np.random.RandomState(seed=0), dissimilarity="precomputed", n_jobs=1, n_init=1)
    embed = mds.fit_transform(x)
    embed = torch.Tensor(embed)

    # scale
    embed = scale_to(embed, batch.y, batch.anchors)

    print("before rotate")
    pred = embed
    loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
    print(f"train:{loss_train.item()}, val:{loss_val.item()}")

    # rotate
    clf = PCA(n_components=2)
    batch.y = torch.Tensor(clf.fit_transform(batch.y))
    embed = torch.Tensor(clf.fit_transform(embed))

    print("after rotate")

    # for epoch in range(100):
    #     model.train()
    #     optimizer.zero_grad()
    #     pred = model(embed)
    #     loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    #     loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
    #     loss_train.backward()
    #     optimizer.step()
    #     print(f"train:{loss_train.item()}, val:{loss_val.item()}")
    pred = embed
    loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
    print(f"train:{loss_train.item()}, val:{loss_val.item()}")

print("fastmode")
loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual")
plt.legend()
plt.title('Manifold')
plt.show()

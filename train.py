from process_dataset import *
from models import *
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np

# data_loader, num_nodes = process_dataset('datasets/comp1_clean.csv', batch_size=1, threshold=1000, fake_links=False)
# num_batches = len(data_loader)
# print(num_batches, "batches")

np.random.seed(0)
torch.manual_seed(0)

num_nodes = 100
num_anchors = 20
threshold = 1.0

data_loader, num_nodes, noisy_distance_matrix = fake_dataset(num_nodes, num_anchors, threshold=threshold)
# data_loader, num_nodes = scoped_dataset(num_nodes, num_anchors, threshold=threshold)
# data_loader, num_nodes, noisy_distance_matrix = modified_adj(num_nodes, num_anchors, threshold=threshold)


# modelname = "GCN"
# modelname = "GCN"
modelname = "GAT"

if modelname == "gfNN":
    model = gfNN(nfeat=num_nodes, nhid=1000, nout=2, dropout=0.5)
elif modelname == "GCN":
    # model = GCN(nfeat=num_nodes, nhid=128, nout=3, dropout=0.01)
    model = GCN(nfeat=num_nodes, nhid=200, nout=2, dropout=0.5)
elif modelname == "GAT":
    model = GAT(nfeat=num_nodes, nhid=200, nout=2, dropout=0.5)
else:
    raise NotImplementedError

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
loss_fn = torch.nn.MSELoss()

wandb_log = False
if wandb_log:
    wandb.init(project="GNN-localize", entity="lillyclark", config={})
    wandb.run.name = "matrix_loss"+"_"+wandb.run.id

start = time.time()
for batch in data_loader:
    if modelname == "gfNN":
        x = torch.sparse.mm(batch.adj, batch.x)
        # x = torch.sparse.mm(batch.adj, x)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        if modelname == "gfNN":
            pred = model(x)
        elif modelname == "GCN" or modelname == "GAT":
            pred = model(batch.x, batch.adj)

        pred_matrix = matrix_from_locs(pred)
        loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
        loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])# + loss_fn(pred_matrix, noisy_distance_matrix)
        loss_train.backward()
        optimizer.step()
        print(f"epoch:{epoch}, train:{loss_train.item()}, val:{loss_val.item()}")

        if wandb_log:
            wandb.log({"loss_train":loss_train})
            wandb.log({"loss_val":loss_val})

print(f"Done in {time.time()-start} seconds.")
if wandb_log:
    wandb.finish()

model.eval()
if modelname == "gfNN":
    x = torch.sparse.mm(batch.adj, batch.x)
    # x = torch.sparse.mm(batch.adj, x)
if modelname == "gfNN":
    pred = model(x)
elif modelname == "GCN" or modelname == "GAT":
    pred = model(batch.x, batch.adj)
loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual")
plt.legend()
plt.title('GCN Localization (GAT)')
plt.show()

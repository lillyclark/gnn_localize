from process_dataset import *
from models import *
from lle_like import solve_like_LLE, neighbors, barycenter_weights, weight_to_mat
from decomposition import normalize_tensor, reduce_rank, denoise_via_SVD
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
threshold = 1.2
n_neighbors = 20

start = time.time()
# data_loader, num_nodes, noisy_distance_matrix = fake_dataset(num_nodes, num_anchors, threshold=threshold)
data_loader, num_nodes, noisy_distance_matrix = nLOS_dataset(num_nodes, num_anchors, threshold=threshold)
# data_loader, num_nodes = scoped_dataset(num_nodes, num_anchors, threshold=threshold)
# data_loader, num_nodes, noisy_distance_matrix = modified_adj(num_nodes, num_anchors, threshold=threshold)


modelname = "GCN"
# modelname = "GAT"
# modelname = "LLE"

print("loaded dataset, using model",modelname)
print(f"{time.time()-start} seconds")

# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.L1Loss()

if modelname == "LLE":
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        reduce_rank = True
        if reduce_rank:
            euclidean_matrix = noisy_distance_matrix**2
            start = time.time()
            noisy_distance_matrix = denoise_via_SVD(euclidean_matrix,k=4,fill_diag=False,take_sqrt=False)
            print(f"{time.time()-start} seconds to do rank reduction")
            start = time.time()
            pred = solve_like_LLE(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=True)
            print(f"{time.time()-start} seconds to solve")
        else:
            start = time.time()
            pred = solve_like_LLE(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=False)
            print(f"{time.time()-start} seconds to solve")
        pred = torch.Tensor(pred)
    loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

else:
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

    wandb_log = False
    if wandb_log:
        wandb.init(project="GNN-localize", entity="lillyclark", config={})
        wandb.run.name = "matrix_loss"+"_"+wandb.run.id

    start = time.time()
    for batch in data_loader:
        if modelname == "gfNN":
            x = torch.sparse.mm(batch.adj, batch.x)
            # x = torch.sparse.mm(batch.adj, x)

        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            if modelname == "gfNN":
                pred = model(x)
            elif modelname == "GCN" or modelname == "GAT":
                pred = model(batch.x, batch.adj)

            # pred_matrix = matrix_from_locs(pred)
            loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
            loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])# + loss_fn(pred_matrix, noisy_distance_matrix)
            loss_train.backward()
            optimizer.step()
            # print(f"epoch:{epoch}, train:{loss_train.item()}, val:{loss_val.item()}")

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

plt.scatter(pred[:num_anchors,0].detach().numpy(), pred[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
plt.scatter(batch.y[:num_anchors,0].detach().numpy(), batch.y[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
plt.scatter(pred[num_anchors:,0].detach().numpy(), pred[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
plt.scatter(batch.y[num_anchors:,0].detach().numpy(), batch.y[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
plt.legend()
plt.title(f"{modelname}: {np.round(torch.sqrt(loss_test).item(),2)}")
plt.show()

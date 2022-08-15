from process_dataset import *
from models import *
from lle_like import solve_like_LLE, solve_like_LLE_anchors, neighbors, barycenter_weights, weight_to_mat, solve_with_LRR
from decomposition import normalize_tensor, reduce_rank, denoise_via_SVD
from separate import *
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

num_nodes = 500
num_anchors = 50
threshold = 1.2
n_neighbors = 25

start = time.time()
data_loader, num_nodes, noisy_distance_matrix = their_dataset(num_nodes, num_anchors, threshold=threshold)
# data_loader, num_nodes, noisy_distance_matrix = fake_dataset(num_nodes, num_anchors, threshold=threshold)
# data_loader, num_nodes, noisy_distance_matrix = nLOS_dataset(num_nodes, num_anchors, threshold=threshold)
# data_loader, num_nodes = scoped_dataset(num_nodes, num_anchors, threshold=threshold)
# data_loader, num_nodes, noisy_distance_matrix = modified_adj(num_nodes, num_anchors, threshold=threshold)


# modelname = "GCN"
# modelname = "GAT"
# modelname = "LLE"
# modelname = "LRR"
modelname = "novel"

print("loaded dataset, using model",modelname)
print(f"{time.time()-start} seconds")


loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.L1Loss()

if modelname == "novel":
    anchors_as_neighbors = False
    assert n_neighbors > 3
    print("low rank plus sparse decomp, then lle-like solve")
    print("n_neighbors =",n_neighbors)
    for batch in data_loader:
        lam = 1/(num_nodes**0.5)*1.1
        mu = 1/(num_nodes**0.5)*1.1
        eps = 0.001
        n_init = 10
        print("lam:",lam)
        print("mu:",mu)
        print("eps:",eps)
        print("n_init:",n_init)

        k1_try = [25000] #[20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000]
        print("try k1:",k1_try)

        start = time.time()
        anchor_locs = batch.y[batch.anchors]

        # k1s = []
        # ffs = []

        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
        best_X, best_Y, best_ff = separate_dataset_multiple_inits(noisy_distance_matrix, k0=4, k1=k1_try[0], n_init=n_init, lam=lam, mu=mu, eps=eps)
        # k1s.append(k1_try[0])
        # ffs.append(best_ff)
        for k1 in k1_try[1:]:
            X, Y, ff = separate_dataset_multiple_inits(noisy_distance_matrix, k0=4, k1=k1, n_init=n_init, lam=lam, mu=mu, eps=eps)
            # k1s.append(k1)
            # ffs.append(ff)
            if ff < best_ff:
                print("k1=",k1)
                best_X, best_Y, best_ff = X, Y, ff
        X, Y, ff = best_X, best_Y, best_ff
        # fig, ax = plt.subplots(1,1)
        # ax.plot(k1s, ffs)
        # ax.set_xlabel("Value of k1 (sparsity)")
        # ax.set_ylabel("Value of f (cost function)")
        # plt.show()

        if anchors_as_neighbors:
            pred = solve_like_LLE_anchors(num_nodes, num_anchors, anchor_locs, X, dont_square=True)
        else:
            pred = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True)
        print(f"{time.time()-start} seconds to solve")
    loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

elif modelname == "LRR":
    assert n_neighbors > 3
    print("local rank reduction")
    print("n_neighbors =",n_neighbors)
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        euclidean_matrix = torch.Tensor(noisy_distance_matrix**2)
        start = time.time()
        pred = solve_with_LRR(num_nodes,num_anchors,n_neighbors,anchor_locs,euclidean_matrix,dont_square=True)
        pred = torch.Tensor(pred)
        print(f"{time.time()-start} seconds to solve")
    loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

elif modelname == "LLE":
    print("n_neighbors =",n_neighbors)
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        reduce_rank = False
        if reduce_rank:
            euclidean_matrix = noisy_distance_matrix**2
            start = time.time()
            noisy_distance_matrix = denoise_via_SVD(euclidean_matrix,k=4,fill_diag=False,take_sqrt=False)
            print(f"{time.time()-start} seconds to do rank reduction")
            start = time.time()
            pred = solve_like_LLE(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=True)
            print(f"{time.time()-start} seconds to solve")
        else:
            noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
            start = time.time()
            pred = solve_like_LLE(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=False)
            print(f"{time.time()-start} seconds to solve")
        pred = torch.Tensor(pred)
    loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

else:
    assert threshold < 5
    if modelname == "gfNN":
        model = gfNN(nfeat=num_nodes, nhid=1000, nout=2, dropout=0.5)
    elif modelname == "GCN":
        # model = GCN(nfeat=num_nodes, nhid=128, nout=3, dropout=0.01)
        model = GCN(nfeat=num_nodes, nhid=2000, nout=2, dropout=0.5)
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

        for epoch in range(200):
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
            if epoch % 20 == 0:
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

plt.scatter(pred[:num_anchors,0].detach().numpy(), pred[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
plt.scatter(batch.y[:num_anchors,0].detach().numpy(), batch.y[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
plt.scatter(pred[num_anchors:,0].detach().numpy(), pred[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
plt.scatter(batch.y[num_anchors:,0].detach().numpy(), batch.y[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
plt.legend()
plt.title(f"{modelname}: {np.round(torch.sqrt(loss_test).item(),2)}")
plt.show()

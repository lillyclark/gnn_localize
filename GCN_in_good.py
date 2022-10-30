from process_dataset import *
from models import *
from lle_like import solve_like_LLE, neighbors, barycenter_weights, weight_to_mat, solve_with_LRR
from decomposition import normalize_tensor, reduce_rank, denoise_via_SVD
from separate import *
import torch.optim as optim
import torch
import wandb
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from experiments import *

if __name__ == "__main__":

    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    print("EXPERIMENT: does GCN work poorly in ideal settings")
    filename = "GCN_in_good.txt"
    figname = "GCN_in_good.pdf"
    figname2 = "GCN_in_good.pdf"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # # GCN PARAMS
    # threshold = 1.2
    # nhid = 2000
    # nout = 2
    # dropout = 0.5
    # lr = 0.01
    # weight_decay = 0
    # num_epochs = 200
    #
    # # NOVEL PARAMS
    # n_neighbors = 50 #25
    # k0 = 4
    # lam = 0.01 #1/(num_nodes**0.5)*1.1
    # mu = 0.1 #1/(num_nodes**0.5)*1.1
    # eps = 0.001
    # n_init = 1
    # k1_init = num_nodes**2*(5/100)
    # step_size = 1
    # eps_k1 = 40000
    # constrain_solution = True

    p_nLOS = 0
    std = 0.0
    noise_floor_dist = None

    start = time.time()
    data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS, std=std, noise_floor_dist=noise_floor_dist)
    print("dataset loaded...")

    print("GCN....")
    model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
    gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
    gcn_total_time = gcn_train_time + gcn_predict_time
    print(f"...done in {gcn_total_time} secs")

    print("GCN ERROR:",gcn_error)

    print("novel...")
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
        start = time.time()
        X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False, constrain_solution=constrain_solution)
        novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
        novel_solve_time = time.time()-start
        novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
        novel_error = torch.sqrt(novel_error).item()
    print(f"...done in {novel_solve_time} secs")

    print("NOVEL ERROR:", novel_error)

    # plot_rmse(figname2, batch.y, {"GCN":gcn_pred, "SMILE":novel_pred})
    # plot_out(figname, batch, gcn_pred, f"GCN ({np.round(gcn_error,2)})", novel_pred, f"SMILE ({np.round(novel_error,2)})", indices=None)

    def make_subplot(ax,test,true_locs,title,other_color=ABLATION_COLOR):
        ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color=TRUE_COLOR)#,alpha=0.1)
        ax.scatter(test[:num_anchors,0].detach().numpy(), test[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color=other_color)
        ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color=TRUE_COLOR)#,alpha=0.1)
        ax.scatter(test[num_anchors:,0].detach().numpy(), test[num_anchors:,1].detach().numpy(), label="predicted",color=other_color)#,alpha=0.1)
        # ax.legend()
        ax.set_title(title)

    fig, axes = plt.subplots(2,2,figsize=(6,6)) #, sharex=True, sharey=True)

    ax = axes[0][0]
    test = gcn_pred
    title = f"GCN ideal ({np.round(gcn_error,2)})"
    make_subplot(ax, test, batch.y, title, other_color=GCN_COLOR)

    ax = axes[0][1]
    test = novel_pred
    title = f"SMILE ideal ({np.round(novel_error,2)})"
    make_subplot(ax, test, batch.y, title, other_color=SMILE_COLOR)

    p_nLOS = 10
    std = 0.5
    noise_floor_dist = None

    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    start = time.time()
    data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS, std=std, noise_floor_dist=noise_floor_dist)
    print("dataset loaded...")

    print("GCN....")
    model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
    gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
    gcn_total_time = gcn_train_time + gcn_predict_time
    print(f"...done in {gcn_total_time} secs")

    print("GCN ERROR:",gcn_error)

    print("novel...")
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
        start = time.time()
        X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False, constrain_solution=constrain_solution)
        novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
        novel_solve_time = time.time()-start
        novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
        novel_error = torch.sqrt(novel_error).item()
    print(f"...done in {novel_solve_time} secs")

    print("NOVEL ERROR:", novel_error)

    ax = axes[1][0]
    test = gcn_pred
    title = f"GCN noisy ({np.round(gcn_error,2)})"
    make_subplot(ax, test, batch.y, title, other_color=GCN_COLOR)

    ax = axes[1][1]
    test = novel_pred
    title = f"SMILE noisy ({np.round(novel_error,2)})"
    make_subplot(ax, test, batch.y, title, other_color=SMILE_COLOR)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)

    fig.tight_layout()
    fig.savefig(figname)
    print("Plot saved to",figname)

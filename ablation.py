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

    print("EXPERIMENT: ablation")
    filename = "ablation.txt"
    figname = "ablation.jpg"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # data params
    p_nLOS = 10
    std = 0.3
    noise_floor_dist = 5.0

    # GCN PARAMS
    threshold = 1.2
    nhid = 2000
    nout = 2
    dropout = 0.5
    lr = 0.01
    weight_decay = 0
    num_epochs = 200

    # NOVEL PARAMS
    n_neighbors = 50
    k0 = 4
    lam = 0.01 #1/(num_nodes**0.5)*1.1
    mu = 0.1 #1/(num_nodes**0.5)*1.1
    eps = 0.001
    n_init = 1
    k1_init = 0 #num_nodes**2*(5/100)
    step_size = 1
    eps_k1 = 40000

    start = time.time()
    print("fake dataset!")
    data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS, std=std, noise_floor_dist=noise_floor_dist)
    # data_loader, num_nodes, noisy_distance_matrix = their_dataset(num_nodes, num_anchors, threshold=threshold)
    print("dataset loaded...")

    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)

        print("Decomposition")
        decomposition = solve_direct(noisy_distance_matrix, anchor_locs, mode="Kabsch")
        decomposition_error = torch.sqrt(loss_fn(decomposition[batch.nodes], batch.y[batch.nodes])).item()

        print("Reduce rank + Decomposition")
        rank_reduced = denoise_via_SVD(noisy_distance_matrix**2, k=4, fill_diag=False, take_sqrt=False)
        rank_reduced_decomposition = solve_direct(rank_reduced, anchor_locs, mode="Kabsch", dont_square=True)
        rank_reduced_decomposition_error = torch.sqrt(loss_fn(rank_reduced_decomposition[batch.nodes], batch.y[batch.nodes])).item()

        print("Sparse inference + Decomposition")
        # X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix**2, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
        k1 = 25000
        X, Y, ff = separate_dataset(noisy_distance_matrix, k0, k1, lam=lam, mu=mu, eps=eps)
        x_decomposition = solve_direct(X, anchor_locs, mode="Kabsch", dont_square=True)
        x_decomposition_error = torch.sqrt(loss_fn(x_decomposition[batch.nodes], batch.y[batch.nodes])).item()

        print("LLE")
        lle = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, noisy_distance_matrix, dont_square=False, anchors_as_neighbors=False, return_indices=False)
        lle_error = torch.sqrt(loss_fn(lle[batch.nodes], batch.y[batch.nodes])).item()

        print("Reduce rank + LLE")
        rank_reduced_lle = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, rank_reduced, dont_square=True, anchors_as_neighbors=False, return_indices=False)
        rank_reduced_lle_error = torch.sqrt(loss_fn(rank_reduced_lle[batch.nodes], batch.y[batch.nodes])).item()

        print("SMILE")
        smile = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, torch.Tensor(X), dont_square=True, anchors_as_neighbors=False, return_indices=False)
        novel_error = torch.sqrt(loss_fn(smile[batch.nodes], batch.y[batch.nodes])).item()
    print("...done")

    def write_():
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')  # Get the Now time
        file_handle = open(filename, mode='a')
        file_handle.write('=====================================\n')
        file_handle.write(nowTime + '\n')
        file_handle.write("num_nodes: " + str(num_nodes) + '\n')
        file_handle.write("num_anchors: " + str(num_anchors) + '\n')

        file_handle.write("p_nLOS: " + str(p_nLOS) + '\n')
        file_handle.write("std: " + str(std) + '\n')
        file_handle.write("noise_floor_dist: " + str(noise_floor_dist) + '\n')

        file_handle.write("threshold: " + str(threshold) + '\n')
        file_handle.write("nhid: " + str(nhid) + '\n')
        file_handle.write("nout: " + str(nout) + '\n')
        file_handle.write("dropout: " + str(dropout) + '\n')
        file_handle.write("lr: " + str(lr) + '\n')
        file_handle.write("weight_decay: " + str(weight_decay) + '\n')
        file_handle.write("num_epochs: " + str(num_epochs) + '\n')

        file_handle.write("n_neighbors: " + str(n_neighbors) + '\n')
        file_handle.write("k0: " +str(k0) + '\n')
        file_handle.write("lam: " +str(np.round(lam,3)) + '\n')
        file_handle.write("mu: " +str(np.round(mu,3)) + '\n')
        file_handle.write("eps: " +str(eps) + '\n')
        file_handle.write("n_init: " +str(n_init) + '\n')
        file_handle.write("k1_init: " +str(k1_init) + '\n')
        file_handle.write("step_size: " +str(step_size) + '\n')
        file_handle.write("eps_k1: " +str(eps_k1) + '\n')
        file_handle.write("k1: " +str(k1) + '\n')

        file_handle.write("RESULTS OF ABLATION" + '\n')
        file_handle.write("decomposition: " + str(np.round(decomposition_error,3)) + '\n')
        file_handle.write("rank_reduced_decomposition: " + str(np.round(rank_reduced_decomposition_error,3)) + '\n')
        file_handle.write("x_decomposition: " + str(np.round(x_decomposition_error,3)) + '\n')
        file_handle.write("lle: " + str(np.round(lle_error,3)) + '\n')
        file_handle.write("rank_reduced_lle: " + str(np.round(rank_reduced_lle_error,3)) + '\n')
        file_handle.write("smile: " + str(np.round(novel_error,3)) + '\n')

        file_handle.close()
        print("Results written to", filename)

    write_()

    def make_subplot(ax,test,true_locs,title):
        ax.scatter(test[:num_anchors,0].detach().numpy(), test[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")
        ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
        ax.scatter(test[num_anchors:,0].detach().numpy(), test[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
        ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
        # ax.legend()
        ax.set_title(title)

    fig, axes = plt.subplots(2,3,figsize=(9,6)) #, sharex=True, sharey=True)

    ax = axes[0][0]
    test = decomposition
    title = f"Decomposition ({np.round(decomposition_error,2)})"
    make_subplot(ax, test, batch.y, title)

    ax = axes[0][1]
    test = rank_reduced_decomposition
    title = f"Reduce Rank + \nDecomposition ({np.round(rank_reduced_decomposition_error,2)})"
    make_subplot(ax, test, batch.y, title)

    ax = axes[0][2]
    test = x_decomposition
    title = f"Sparse Inference + \nDecomposition ({np.round(x_decomposition_error,2)})"
    make_subplot(ax, test, batch.y, title)

    ax = axes[1][0]
    test = lle
    title = f"LLE ({np.round(lle_error,2)})"
    make_subplot(ax, test, batch.y, title)

    ax = axes[1][1]
    test = rank_reduced_lle
    title = f"Reduce Rank + \nLLE ({np.round(rank_reduced_lle_error,2)})"
    make_subplot(ax, test, batch.y, title)

    ax = axes[1][2]
    test = smile
    title = f"SMILE ({np.round(novel_error,2)})"
    make_subplot(ax, test, batch.y, title)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)

    fig.tight_layout()
    fig.savefig(figname)
    print("Plot saved to",figname)

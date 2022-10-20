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
    print("EXPERIMENT: k1 and objective function")

    figname = "k1_and_convergence.pdf"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    percent_nLOS_list = [10,20,30,40,50]
    colors = ['red','orange','yellow','green','blue','pink']

    fig, ax = plt.subplots(1,1,figsize=(6,3))

    for i, p_nLOS in enumerate(percent_nLOS_list):
        print("p_nLOS:",p_nLOS)

        k1_init = 0
        step_size = 1
        data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS, std=std, noise_floor_dist=noise_floor_dist)

        for batch in data_loader:
            anchor_locs = batch.y[batch.anchors]
            noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
            start = time.time()

            k1s = []
            fs = []

            measured = noisy_distance_matrix
            num_edges = int(measured.shape[0]*measured.shape[1])
            step_size = int(num_edges*step_size/100)
            k1 = k1_init
            X, Y, fi = separate_dataset_multiple_inits(measured, k0, k1, n_init=n_init, lam=lam, mu=mu, eps=eps, constrain_solution=constrain_solution)
            k1s.append(k1)
            fs.append(fi)
            for iter in range(60):
                k1 += step_size
                X, Y, ff = separate_dataset_multiple_inits(measured, k0, int(k1), n_init=n_init, lam=lam, mu=mu, eps=eps, constrain_solution=constrain_solution)
                k1s.append(k1)
                fs.append(ff)
                fi = ff

        ax.plot(k1s, fs, color=colors[i])
        ax.axvline(true_k1, color=colors[i], linestyle='--')

    ax.set_xlabel(r'$\hat{\beta}$')
    ax.set_ylabel(r'Objective function $f$')
    fig.tight_layout()
    fig.savefig(figname)

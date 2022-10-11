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

    print("EXPERIMENT: vary mu and lambda")
    filename = "vary_mu_lambda.txt"
    figname = "vary_mu_lambda.jpg"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # NOVEL PARAMS
    n_neighbors = 25
    k0 = 4
    # lam = 1/(num_nodes**0.5)*1.1
    # lam_list = [0.03,0.04,0.05,0.06,0.07]
    lam_list = [0.0,0.01,0.02,0.03]
    # mu = 1/(num_nodes**0.5)*1.1
    mu_list = [0.07,0.08,0.09,0.1]
    eps = 0.001
    n_init = 1
    k1_init = num_nodes**2*(5/100)
    step_size = 1
    eps_k1 = 40000

    data_loader, num_nodes, noisy_distance_matrix = their_dataset(num_nodes, num_anchors, threshold=10)
    print("dataset loaded...")

    def writeA():
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')  # Get the Now time
        file_handle = open(filename, mode='a')
        file_handle.write('=====================================\n')
        file_handle.write(nowTime + '\n')
        file_handle.write("num_nodes: " + str(num_nodes) + '\n')
        file_handle.write("num_anchors: " + str(num_anchors) + '\n')
        file_handle.write("n_neighbors: " + str(n_neighbors) + '\n')
        file_handle.write("k0: " +str(k0) + '\n')
        # file_handle.write("lam: " +str(np.round(lam,3)) + '\n')
        # file_handle.write("mu: " +str(np.round(mu,3)) + '\n')
        file_handle.write("eps: " +str(eps) + '\n')
        file_handle.write("n_init: " +str(n_init) + '\n')
        file_handle.write("k1_init: " +str(k1_init) + '\n')
        file_handle.write("step_size: " +str(step_size) + '\n')
        file_handle.write("eps_k1: " +str(eps_k1) + '\n')
        file_handle.write("lambda list: " +str(lam_list) + '\n')
        file_handle.write("mu list: " +str(mu_list) + '\n')
        file_handle.close()
        print("Overview written to", filename)

    def writeLine(value, end=False):
        file_handle = open(filename, mode='a')
        if end:
            file_handle.write(str(value)+"\n")
        else:
            file_handle.write(str(value)+", ")
        file_handle.close()

    writeA()

    print("novel...")
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)

        result_arr = np.zeros((len(lam_list),len(mu_list)))

        for i, lam in enumerate(lam_list):
            for j, mu in enumerate(mu_list):
                start = time.time()
                # k1, novel_error, novel_solve_time = np.random.random(), np.random.random(), np.random.random()
                X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, plot=False)
                novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
                novel_solve_time = time.time()-start
                novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
                novel_error = torch.sqrt(novel_error).item()
                result_arr[i][j] = novel_error
                print("lam:",lam,"mu:",mu,"k1:",k1,"error:",novel_error,"runtime:",novel_solve_time)
    print("...done")

    writeLine("Errors:"+str(result_arr), True)

    fig, ax1 = plt.subplots(figsize=(3,3))
    a = ax1.imshow(result_arr,origin='lower')
    plt.colorbar(a, ax=ax1, shrink=0.6,aspect=10)
    ax1.set_xticks(np.arange(len(mu_list)))
    ax1.set_xticklabels(mu_list)
    ax1.set_xlabel(r"$\mu$")
    ax1.set_yticks(np.arange(len(lam_list)))
    ax1.set_yticklabels(lam_list)
    ax1.set_ylabel(r"$\lambda$")
    plt.title("RMSE")

    plt.tight_layout()
    plt.savefig(figname)
    print("plot saved to",figname)

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

    print("EXPERIMENT: vary n_neighbors")
    filename = "vary_n_neighbors.txt"
    figname = "vary_n_neighbors.pdf"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # NOVEL PARAMS
    # n_neighbor_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    n_neighbor_list = [6,8,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
    # n_neighbor_list = [3,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,499]
    n_init = 1
    # k0 = 4
    # lam = 1/(num_nodes**0.5)*1.1
    # mu = 1/(num_nodes**0.5)*1.1
    # eps = 0.001
    # n_init = 1
    # k1_init = num_nodes**2*(5/100)
    # step_size = 1
    # eps_k1 = 40000

    data_loader, num_nodes, noisy_distance_matrix = their_dataset(num_nodes, num_anchors, threshold=10)
    print("dataset loaded...")

    def writeA():
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')  # Get the Now time
        file_handle = open(filename, mode='a')
        file_handle.write('=====================================\n')
        file_handle.write(nowTime + '\n')
        file_handle.write("num_nodes: " + str(num_nodes) + '\n')
        file_handle.write("num_anchors: " + str(num_anchors) + '\n')
        file_handle.write("k0: " +str(k0) + '\n')
        file_handle.write("lam: " +str(np.round(lam,3)) + '\n')
        file_handle.write("mu: " +str(np.round(mu,3)) + '\n')
        file_handle.write("eps: " +str(eps) + '\n')
        file_handle.write("n_init: " +str(n_init) + '\n')
        file_handle.write("k1_init: " +str(k1_init) + '\n')
        file_handle.write("step_size: " +str(step_size) + '\n')
        file_handle.write("eps_k1: " +str(eps_k1) + '\n')
        file_handle.write("n_neighbor_list: " +str(n_neighbor_list) + '\n')
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
        result_list = []
        k1_list = []
        runtime_list = []
        for n_neighbors in n_neighbor_list:
            anchor_locs = batch.y[batch.anchors]
            noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)

            e = []
            r = []
            for test in range(10):
                start = time.time()
                X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, plot=False)
                novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
                novel_solve_time = time.time()-start
                novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
                novel_error = torch.sqrt(novel_error).item()
                e.append(novel_error)
                r.append(novel_solve_time)

            result_list.append(sum(e)/len(e))
            runtime_list.append(sum(r)/len(r))
            k1_list.append(k1)
            print("neighbors:",n_neighbors,"k1:",k1,"error:",novel_error,"runtime:",novel_solve_time)
    print("...done")

    writeLine("Errors:"+str(result_list), True)
    writeLine("k1 estimates:"+str(k1_list), True)
    writeLine("Runtimes:"+str(runtime_list), True)

    fig, ax1 = plt.subplots(figsize=(6,3))

    ax1.plot(n_neighbor_list, result_list, marker='o',color=SMILE_COLOR)
    ax1.set_xlabel('Number of neighbors')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(n_neighbor_list, runtime_list, marker='X', color=SMILE_COLOR, linestyle='--')
    ax2.set_ylabel('Runtime (sec)')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.tight_layout()
    plt.savefig(figname)
    print("plot saved to",figname)

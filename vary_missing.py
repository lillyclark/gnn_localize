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

    print("EXPERIMENT: vary missing")
    filename = "vary_missing.txt"
    figname = "vary_missing.jpg"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # DATA PARAMS
    max_dist_list = [3,4,5,6,7,8]
    p_nLOS = 10
    std = 0.1

    # GCN PARAMS
    threshold = 1.2
    nhid = 2000
    nout = 2
    dropout = 0.5
    lr = 0.01
    weight_decay = 0
    num_epochs = 200

    # NOVEL PARAMS
    n_neighbors = 15
    k0 = 4
    lam = 0.01 #1/(num_nodes**0.5)*1.1
    mu = 0.1 #1/(num_nodes**0.5)*1.1
    eps = 0.001
    n_init = 10

    k1_init = 0 #num_nodes**2*(0/100)
    step_size = 1
    eps_k1 = 40000

    def writeA():
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')  # Get the Now time
        file_handle = open(filename, mode='a')
        file_handle.write('=====================================\n')
        file_handle.write(nowTime + '\n')
        file_handle.write("num_nodes: " + str(num_nodes) + '\n')
        file_handle.write("num_anchors: " + str(num_anchors) + '\n')

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

        file_handle.write("p_nLOS: " +str(p_nLOS) + '\n')
        file_handle.write("std: " +str(std) + '\n')
        file_handle.write("max_dist_list: " +str(max_dist_list) + '\n')
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

    novel_list = []
    gcn_list = []
    true_k1_list = []
    k1_estimate_list = []

    for max_dist in max_dist_list:
        print("std:",std)
        data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS, std=std, noise_floor_dist=max_dist)
        true_k1_list.append(true_k1)

        print("GCN....")
        model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
        gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
        gcn_total_time = gcn_train_time + gcn_predict_time
        print("...done")
        gcn_list.append(gcn_error)
        print("gcn error:",gcn_list)

        print("novel...")
        for batch in data_loader:
            anchor_locs = batch.y[batch.anchors]
            noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
            start = time.time()
            print("***set k1_init***")
            k1_init = true_k1 - 5000
            X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
            # print("****to debug, assume k1 known****")
            # k1 = true_k1+100 #p_nLOS*(num_nodes**2)
            # X, Y, ff = separate_dataset_multiple_inits(noisy_distance_matrix, k0, k1, n_init=n_init, lam=lam, mu=mu, eps=eps)
            k1_estimate_list.append(k1)
            novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
            novel_solve_time = time.time()-start
            novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
            novel_error = torch.sqrt(novel_error).item()
        print("...done")
        novel_list.append(novel_error)
        print("novel error:", novel_list)

    writeLine("GCN error:"+str(gcn_list),True)
    writeLine("True K1s:"+str(true_k1_list),True)
    writeLine("K1 estimates:"+str(k1_estimate_list),True)
    writeLine("Novel error:"+str(novel_list),True)

    fig, ax1 = plt.subplots(figsize=(6,3))
    ax1.plot(max_dist_list, novel_list, marker='o',color='blue',label="SMILE")
    ax1.plot(max_dist_list, gcn_list, marker='o',color='orange',label="GCN")
    ax1.set_xlabel('Communication Radius')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    plt.tight_layout()
    plt.savefig(figname)
    print("plot saved to",figname)
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

    print("EXPERIMENT: robustness: vary percentage nLOS, std noise, and noise_floor_dist")
    filename = "robustness.txt"
    figname = "robustness.pdf"
    loss_fn = torch.nn.MSELoss()

    n_init = 2

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

    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(6,6))

    print("vary nLOS...")
    percent_nLOS_list = [0,5,10,15,20,25,30,35,40,45,50]
    k1_init = 0 #num_nodes**2*(0/100)

    novel_list = []
    gcn_list = []

    for p_nLOS in percent_nLOS_list:
        print("p_nLOS:",p_nLOS)
        data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS)

        print("GCN....")
        model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
        gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
        gcn_total_time = gcn_train_time + gcn_predict_time
        print("...done",gcn_total_time)
        gcn_list.append(gcn_error)
        print("gcn error:",gcn_error)

        print("novel...")
        for batch in data_loader:
            anchor_locs = batch.y[batch.anchors]
            noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
            start = time.time()
            X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
            novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
            novel_solve_time = time.time()-start
            novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
            novel_error = torch.sqrt(novel_error).item()
        print("...done",novel_solve_time)
        novel_list.append(novel_error)
        print("true k1:",true_k1,"est:",k1)
        print("novel error:", novel_error)

    writeLine("percent_nLOS_list:"+str(percent_nLOS_list),True)
    writeLine("GCN error:"+str(gcn_list),True)
    writeLine("Novel error:"+str(novel_list),True)

    ax1.plot(percent_nLOS_list, novel_list, marker='o',color=SMILE_COLOR,label="SMILE")
    ax1.plot(percent_nLOS_list, gcn_list, marker='o',color=GCN_COLOR,label="GCN")
    ax1.set_xlabel(r'$p_{NLOS}$')
    ax1.set_ylabel('RMSE')
    ax1.legend()


    print("vary noise...")
    std_list = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    p_nLOS = 10

    novel_list = []
    gcn_list = []

    for std in std_list:
        print("std:",std)
        data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS, std=std)

        print("GCN....")
        model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
        gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
        gcn_total_time = gcn_train_time + gcn_predict_time
        print("...done",gcn_total_time)
        gcn_list.append(gcn_error)
        print("gcn error:",gcn_error)

        print("novel...")
        for batch in data_loader:
            anchor_locs = batch.y[batch.anchors]
            noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
            start = time.time()
            X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
            novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
            novel_solve_time = time.time()-start
            novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
            novel_error = torch.sqrt(novel_error).item()
        print("...done",novel_solve_time)
        novel_list.append(novel_error)
        print("true k1:",true_k1,"est:",k1)
        print("novel error:", novel_error)

    writeLine("std_list:"+str(std_list),True)
    writeLine("GCN error:"+str(gcn_list),True)
    writeLine("Novel error:"+str(novel_list),True)

    ax2.plot(std_list, novel_list, marker='o',color=SMILE_COLOR,label="SMILE")
    ax2.plot(std_list, gcn_list, marker='o',color=GCN_COLOR,label="GCN")
    ax2.set_xlabel(r'$\sigma$')
    ax2.set_ylabel('RMSE')
    ax2.legend()

    print("vary missing...")
    std = 0.1
    max_dist_list = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8]

    novel_list = []
    gcn_list = []

    for max_dist in max_dist_list:
        print("max_dist:",max_dist)
        data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS, std=std, noise_floor_dist=max_dist)

        print("GCN....")
        model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
        gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
        gcn_total_time = gcn_train_time + gcn_predict_time
        print("...done",gcn_total_time)
        gcn_list.append(gcn_error)
        print("gcn error:",gcn_error)

        print("novel...")
        for batch in data_loader:
            anchor_locs = batch.y[batch.anchors]
            noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
            start = time.time()
            X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
            novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
            novel_solve_time = time.time()-start
            novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
            novel_error = torch.sqrt(novel_error).item()
        print("...done",novel_solve_time)
        novel_list.append(novel_error)
        print("true k1:",true_k1,"est:",k1)
        print("novel error:", novel_error)

    writeLine("max_dist_list:"+str(max_dist_list),True)
    writeLine("GCN error:"+str(gcn_list),True)
    writeLine("Novel error:"+str(novel_list),True)

    ax3.plot(max_dist_list, novel_list, marker='o',color=SMILE_COLOR,label="SMILE")
    ax3.plot(max_dist_list, gcn_list, marker='o',color=GCN_COLOR,label="GCN")
    ax3.set_xlabel(r'$\theta$')
    ax3.set_ylabel('RMSE')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(figname)
    print("plot saved to",figname)

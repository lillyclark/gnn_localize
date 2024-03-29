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

if __name__=="__main__":
    print("EXPERIMENT: CELLPHONE DATA")
    filename = "cellphone.txt"
    figname = "cellphone.pdf"
    figname2 = "cellphone_rmse.pdf"

    num_anchors = 4
    loss_fn = torch.nn.MSELoss()

    # RSS TO DIST PARAMS
    eta = -2.9032366
    Kref = -46.120605

    # # GCN PARAMS
    threshold = 5.2

    # NOVEL PARAMS
    n_init = 10
    n_neighbors = 3
    lam = 0.05
    mu = 0.05
    eps = 0.001
    step_size = 1
    eps_k1 = 0.01
    constrain_solution = False

    data_loader, num_nodes, noisy_distance_matrix = load_cellphone_data(num_anchors=num_anchors, threshold=threshold)
    print("dataset loaded...")

    avg_gcn_error = []
    avg_gcn_time = []
    avg_novel_error = []
    avg_novel_time = []

    def write_():
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
        file_handle.write("k1: " +str(k1) + '\n')
        file_handle.write("lam: " +str(np.round(lam,3)) + '\n')
        file_handle.write("mu: " +str(np.round(mu,3)) + '\n')
        file_handle.write("eps: " +str(eps) + '\n')
        file_handle.write("n_init: " +str(n_init) + '\n')
        file_handle.write("k1_init: " +str(k1_init) + '\n')
        file_handle.write("step_size: " +str(step_size) + '\n')
        file_handle.write("eps_k1: " +str(eps_k1) + '\n')

        file_handle.write("RESULTS OF EXPERIMENT 3" + '\n')
        file_handle.write("GCN RMSE: " + str(np.round(gcn_error,3)) + '\n')
        file_handle.write("GCN total time:" + str(np.round(gcn_total_time,3)) + '\n')
        file_handle.write("Novel RMSE: " + str(np.round(novel_error,3)) + '\n')
        file_handle.write("Novel solve time:" + str(np.round(novel_solve_time,3)) + '\n')

        file_handle.close()
        print("Results written to", filename)

    for seed_ in range(10):
        print("seed_:",seed_)

        np.random.seed(seed_)
        torch.manual_seed(seed_)

        print("GCN....")
        model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
        gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
        gcn_total_time = gcn_train_time + gcn_predict_time
        print("gcn RMSE:",gcn_error)
        print("...done")

        print("novel...")
        for batch in data_loader:
            anchor_locs = batch.y[batch.anchors]
            start = time.time()
            # X, Y, ff = separate_dataset_multiple_inits(noisy_distance_matrix, k0=k0, k1=k1, n_init=n_init, lam=lam, mu=mu, eps=eps)
            X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
            print("k1:",k1)
            novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
            novel_solve_time = time.time()-start
            novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
            novel_error = torch.sqrt(novel_error).item()
            print("novel RMSE:",novel_error)
        print("...done")

        avg_gcn_error.append(gcn_error)
        avg_gcn_time.append(gcn_total_time)
        avg_novel_error.append(novel_error)
        avg_novel_time.append(novel_solve_time)

    gcn_error = sum(avg_gcn_error)/len(avg_gcn_error)
    gcn_total_time = sum(avg_gcn_time)/len(avg_gcn_time)
    novel_error = sum(avg_novel_error)/len(avg_novel_error)
    novel_solve_time = sum(avg_novel_time)/len(avg_novel_time)

    print("GCN:",gcn_error,"SMILE:",novel_error)

    write_()

    plot_rmse(figname2, batch.y, {"GCN":gcn_pred, "SMILE":novel_pred})
    plot_out(figname, batch, gcn_pred, f"GCN ({np.round(gcn_error,2)})", novel_pred, f"SMILE ({np.round(novel_error,2)})", indices=indices)

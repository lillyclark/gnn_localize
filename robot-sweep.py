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
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    print("EXPERIMENT: ROBOT DATA -- SWEEP")
    filename = "robot_sweep.txt"

    num_anchors = 13
    loss_fn = torch.nn.MSELoss()

    # RSS TO DIST PARAMS
    eta = 4.57435973
    Kref = 13.111276899657597

    BEST_GCN_RMSE = 1000
    BEST_SMILE_RMSE = 1000

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
        # file_handle.write("k1: " +str(k1) + '\n')
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

    # NOVEL PARAMS
    n_init = 5
    n_neighbors = 10

    # for threshold in [120,121,122,123,124,125,126,127,128,129,130]:

    for n_neighbors in [5,7,9,11,13]:
        print("n_neighbors:",n_neighbors)
        for lam in [0.01, 0.05, 0.1, 0.5]:
            for mu in [0.01, 0.05, 0.1, 0.5]:
                for eps in [0.0001, 0.001, 0.01, 0.1]:
                    for step_size in [0.01, 0.05, 0.1, 0.5, 1, 2]:
                        for eps_k1 in [0.0001, 0.001, 0.01, 0.1]:
                            for constrain_solution in [False]: #[True, False]:

                                # GCN PARAMS
                                threshold = 124

                                data_loader, num_nodes, noisy_distance_matrix = load_a_moment(eta=eta, Kref=Kref, threshold=threshold, plot=False, verbose=False, augment=True)

                                model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
                                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                                gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
                                gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
                                gcn_total_time = gcn_train_time + gcn_predict_time

                                for batch in data_loader:
                                    anchor_locs = batch.y[batch.anchors]
                                    start = time.time()
                                    # X, Y, ff = separate_dataset_multiple_inits(noisy_distance_matrix, k0=k0, k1=k1, n_init=n_init, lam=lam, mu=mu, eps=eps)
                                    X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
                                    novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
                                    novel_solve_time = time.time()-start
                                    novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
                                    novel_error = torch.sqrt(novel_error).item()

                                # if gcn_error < BEST_GCN_RMSE:
                                #     print("best threshold:",threshold)
                                #     BEST_GCN_RMSE = gcn_error
                                #     write_()

                                if novel_error < BEST_SMILE_RMSE:
                                    print("best_combo:")
                                    print("n_neighbors:",n_neighbors, "lam:",lam, "mu:", mu, "eps:", eps, "step_size:", step_size, "eps_k1:", eps_k1, "constrain_solution:",constrain_solution)
                                    BEST_SMILE_RMSE = novel_error
                                    write_()

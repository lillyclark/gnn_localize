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

    print("EXPERIMENT: their dataset")
    filename = "compare_with_random.txt"
    figname = "compare_with_random.jpg"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # GCN PARAMS
    threshold = 1.2
    nhid = 2000
    nout = 2
    dropout = 0.5
    lr = 0.01
    weight_decay = 0
    num_epochs = 200

    # NOVEL PARAMS
    n_neighbors = 50#25
    k0 = 4
    lam = 0.01 #1/(num_nodes**0.5)*1.1
    mu = 0.1 #1/(num_nodes**0.5)*1.1
    eps = 0.001
    n_init = 1
    k1_init = num_nodes**2*(5/100)
    step_size = 1
    eps_k1 = 40000

    start = time.time()
    data_loader, num_nodes, noisy_distance_matrix = their_dataset(num_nodes, num_anchors, threshold=threshold)
    print("dataset loaded...")

    print("GCN....")
    model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
    gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
    gcn_total_time = gcn_train_time + gcn_predict_time
    print("...done")

    print("GCN ERROR:",gcn_error)

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
    print("...done")

    print("NOVEL ERROR:", novel_error)

    print("Random...")
    start = time.time()
    random_pred = torch.rand((num_nodes-num_anchors,2))*5
    random_pred_sort, _ = torch.sort(random_pred,dim=1)
    random_pred_sort, _ = torch.sort(random_pred_sort,dim=0)

    true_locs = batch.y[batch.nodes]
    true_locs_sort, _ = torch.sort(true_locs,dim=1)
    true_locs_sort, _ = torch.sort(true_locs_sort,dim=0)
    random_solve_time = time.time()-start

    random_error = loss_fn(random_pred_sort, true_locs_sort)
    random_error = torch.sqrt(random_error).item()

    print("RANDOM ERROR:", random_error)

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
        file_handle.write("lam: " +str(np.round(lam,3)) + '\n')
        file_handle.write("mu: " +str(np.round(mu,3)) + '\n')
        file_handle.write("eps: " +str(eps) + '\n')
        file_handle.write("n_init: " +str(n_init) + '\n')
        file_handle.write("k1_init: " +str(k1_init) + '\n')
        file_handle.write("step_size: " +str(step_size) + '\n')
        file_handle.write("eps_k1: " +str(eps_k1) + '\n')

        file_handle.write("RESULTS OF EXPERIMENT 1" + '\n')
        file_handle.write("GCN RMSE: " + str(np.round(gcn_error,3)) + '\n')
        file_handle.write("GCN train time:" + str(np.round(gcn_train_time,3)) + '\n')
        file_handle.write("GCN predict time:" + str(np.round(gcn_predict_time,3)) + '\n')
        file_handle.write("GCN total time:" + str(np.round(gcn_total_time,3)) + '\n')

        file_handle.write("Novel RMSE: " + str(np.round(novel_error,3)) + '\n')
        file_handle.write("Novel k1 estimate: " + str(k1) + '\n')
        file_handle.write("Novel total time:" + str(np.round(novel_solve_time,3)) + '\n')

        file_handle.write("Random RMSE: " + str(np.round(random_error,3)) + '\n')
        file_handle.write("Random total time:" + str(np.round(random_solve_time,3)) + '\n')

        file_handle.close()
        print("Results written to", filename)

    write_()

    def plot_out(figname, batch, left_pred, left_title, middle_pred, middle_title, right_pred, right_title, indices=None):
        left_actual = batch.y
        right_actual = batch.y
        middle_actual = batch.y
        num_anchors = torch.sum(batch.anchors)
        fig, (left, middle, right) = plt.subplots(1,3,figsize=(12,4), sharex=True, sharey=True)
        left.scatter(left_pred[:num_anchors,0].detach().numpy(), left_pred[:num_anchors,1].detach().numpy(), label="pred anchor", marker="+",color="blue")
        left.scatter(left_actual[:num_anchors,0].detach().numpy(), left_actual[:num_anchors,1].detach().numpy(), label="true anchor", marker="x",color="orange")
        left.scatter(left_pred[num_anchors:,0].detach().numpy(), left_pred[num_anchors:,1].detach().numpy(), label="pred node",color="blue")
        left.scatter(left_actual[num_anchors:,0].detach().numpy(), left_actual[num_anchors:,1].detach().numpy(), label="true node",color="orange")
        left.set_title(left_title)

        middle.scatter(middle_pred[:num_anchors,0].detach().numpy(), middle_pred[:num_anchors,1].detach().numpy(), label="pred anchor", marker="+",color="blue")
        middle.scatter(middle_actual[:num_anchors,0].detach().numpy(), middle_actual[:num_anchors,1].detach().numpy(), label="true anchor", marker="x",color="orange")
        middle.scatter(middle_pred[num_anchors:,0].detach().numpy(), middle_pred[num_anchors:,1].detach().numpy(), label="pred node",color="blue")
        middle.scatter(middle_actual[num_anchors:,0].detach().numpy(), middle_actual[num_anchors:,1].detach().numpy(), label="true node",color="orange")
        middle.set_title(middle_title)

        right.scatter(right_pred[:num_anchors,0].detach().numpy(), right_pred[:num_anchors,1].detach().numpy(), label="pred anchor", marker="+",color="blue")
        right.scatter(right_actual[:num_anchors,0].detach().numpy(), right_actual[:num_anchors,1].detach().numpy(), label="true anchor", marker="x",color="orange")
        right.scatter(right_pred[num_anchors:,0].detach().numpy(), right_pred[num_anchors:,1].detach().numpy(), label="pred node",color="blue")
        right.scatter(right_actual[num_anchors:,0].detach().numpy(), right_actual[num_anchors:,1].detach().numpy(), label="true node",color="orange")
        right.set_title(right_title)
        handles, labels = right.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4)
        fig.tight_layout()
        fig.savefig(figname)
        print("Plot saved to",figname)

    plot_out(figname, batch, gcn_pred, "GCN (Yan et al.)", novel_pred, "SMILE", random_pred, "Random", indices=indices)

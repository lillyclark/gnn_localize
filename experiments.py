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

def train_GCN(model, optimizer, loss_fn, data_loader, num_epochs):
    start = time.time()
    for batch in data_loader:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(batch.x, batch.adj)
            # loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
            loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
            loss_train.backward()
            optimizer.step()
    return time.time()-start

def test_GCN(model, loss_fn, data_loader):
    start = time.time()
    for batch in data_loader:
        model.eval()
        pred = model(batch.x, batch.adj)
        loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
        loss_test = torch.sqrt(loss_test).item()
    return pred, loss_test, time.time()-start

def plot_out(figname, left_pred, left_actual, left_title, right_pred, right_actual, right_title, num_anchors):
    fig, (left, right) = plt.subplots(1,2,figsize=(8,4))
    left.scatter(left_pred[:num_anchors,0].detach().numpy(), left_pred[:num_anchors,1].detach().numpy(), label="pred anchor", marker="+",color="blue")
    left.scatter(left_actual[:num_anchors,0].detach().numpy(), left_actual[:num_anchors,1].detach().numpy(), label="true anchor", marker="x",color="orange")
    left.scatter(left_pred[num_anchors:,0].detach().numpy(), left_pred[num_anchors:,1].detach().numpy(), label="pred node",color="blue")
    left.scatter(left_actual[num_anchors:,0].detach().numpy(), left_actual[num_anchors:,1].detach().numpy(), label="true node",color="orange")
    left.set_title(left_title)
    # left.legend()
    right.scatter(right_pred[:num_anchors,0].detach().numpy(), right_pred[:num_anchors,1].detach().numpy(), label="pred anchor", marker="+",color="blue")
    right.scatter(right_actual[:num_anchors,0].detach().numpy(), right_actual[:num_anchors,1].detach().numpy(), label="true anchor", marker="x",color="orange")
    right.scatter(right_pred[num_anchors:,0].detach().numpy(), right_pred[num_anchors:,1].detach().numpy(), label="pred node",color="blue")
    right.scatter(right_actual[num_anchors:,0].detach().numpy(), right_actual[num_anchors:,1].detach().numpy(), label="true node",color="orange")
    right.set_title(right_title)
    # right.legend()
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.tight_layout()
    fig.savefig(figname)
    print("Plot saved to",figname)

def experiment1():
    print("EXPERIMENT 1: their dataset")
    filename = "experiment1.txt"
    figname = "experiment1.jpg"
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
    n_neighbors = 25
    k0 = 4
    lam = 1/(num_nodes**0.5)*1.1
    mu = 1/(num_nodes**0.5)*1.1
    eps = 0.001
    n_init = 1
    k1_init = num_nodes**2*(5/100)
    step_size = 1

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

    print("novel...")
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
        start = time.time()
        X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, plot=False)
        novel_pred = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False)
        novel_solve_time = time.time()-start
        start = time.time()
        novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
        novel_error = torch.sqrt(novel_error).item()
        novel_predict_time = time.time()-start
        novel_total_time = novel_solve_time + novel_predict_time
    print("...done")

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

        file_handle.write("RESULTS OF EXPERIMENT 1" + '\n')
        file_handle.write("GCN RMSE: " + str(np.round(gcn_error,3)) + '\n')
        file_handle.write("GCN train time:" + str(np.round(gcn_train_time,3)) + '\n')
        file_handle.write("GCN predict time:" + str(np.round(gcn_predict_time,3)) + '\n')
        file_handle.write("GCN total time:" + str(np.round(gcn_total_time,3)) + '\n')
        file_handle.write("Novel RMSE: " + str(np.round(novel_error,3)) + '\n')
        file_handle.write("Novel k1 estimate: " + str(k1) + '\n')
        file_handle.write("Novel solve time:" + str(np.round(novel_solve_time,3)) + '\n')
        file_handle.write("Novel predict time:" + str(np.round(novel_predict_time,3)) + '\n')
        file_handle.write("Novel total time:" + str(np.round(novel_total_time,3)) + '\n')

        file_handle.close()
        print("Results written to", filename)
    write_()

    plot_out(figname, gcn_pred, batch.y, "GCN (Yan et al.)", novel_pred, batch.y, "Novel", num_anchors)

def experiment2():
    print("EXPERIMENT 2: ROBOT DATA")
    filename = "experiment2.txt"
    figname = "experiment2.jpg"
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # RSS TO DIST PARAMS
    eta = 4.11707152
    Kref = 25.110978914824088

    # GCN PARAMS
    threshold = 33
    nhid = 2000
    nout = 2
    dropout = 0.5
    lr = 0.01
    weight_decay = 0
    num_epochs = 200

    data_loader, num_nodes, noisy_distance_matrix = load_a_moment(eta=eta, Kref=Kref, threshold=threshold)
    print("dataset loaded...")

    # NOVEL PARAMS
    n_neighbors = 8
    k0 = 4
    k1 = 136
    lam = 1/(num_nodes**0.5)
    mu = 1/(num_nodes**0.5)
    eps = 0.001
    n_init = 10

    print("GCN....")
    model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs)
    gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
    gcn_total_time = gcn_train_time + gcn_predict_time
    print("...done")

    print("novel...")
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        num_anchors = anchor_locs.shape[0]
        start = time.time()
        X, Y, ff = separate_dataset_multiple_inits(noisy_distance_matrix, k0=k0, k1=k1, n_init=n_init, lam=lam, mu=mu, eps=eps)
        novel_pred = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False)
        novel_solve_time = time.time()-start
        start = time.time()
        novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
        novel_error = torch.sqrt(novel_error).item()
        novel_predict_time = time.time()-start
        novel_total_time = novel_solve_time + novel_predict_time
    print("...done")

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
        file_handle.write("k1: " +str(k0) + '\n')
        file_handle.write("lam: " +str(np.round(lam,3)) + '\n')
        file_handle.write("mu: " +str(np.round(mu,3)) + '\n')
        file_handle.write("eps: " +str(eps) + '\n')
        file_handle.write("n_init: " +str(n_init) + '\n')

        file_handle.write("RESULTS OF EXPERIMENT 2" + '\n')
        file_handle.write("GCN RMSE: " + str(np.round(gcn_error,3)) + '\n')
        file_handle.write("GCN train time:" + str(np.round(gcn_train_time,3)) + '\n')
        file_handle.write("GCN predict time:" + str(np.round(gcn_predict_time,3)) + '\n')
        file_handle.write("GCN total time:" + str(np.round(gcn_total_time,3)) + '\n')
        file_handle.write("Novel RMSE: " + str(np.round(novel_error,3)) + '\n')
        file_handle.write("Novel solve time:" + str(np.round(novel_solve_time,3)) + '\n')
        file_handle.write("Novel predict time:" + str(np.round(novel_predict_time,3)) + '\n')
        file_handle.write("Novel total time:" + str(np.round(novel_total_time,3)) + '\n')

        file_handle.close()
        print("Results written to", filename)
    write_()

    plot_out(figname, gcn_pred, batch.y, "GCN (Yan et al.)", novel_pred, batch.y, "Novel", num_anchors)


if __name__ == "__main__":
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    # experiment1()
    experiment2()

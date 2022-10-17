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

def plot_rmse(figname, true_locs, to_eval):
    plot_data = []
    colors = []
    names = []
    color_list = ["blue","orange","green","red","purple","yellow"]
    fig, ax = plt.subplots(1,1,figsize=(6,3))
    for name, pred_locs in to_eval.items():
        rmse = np.linalg.norm(pred_locs.detach().numpy() - true_locs.detach().numpy(), axis=1)
        plot_data.append(rmse)
        colors.append(color_list[len(colors)])
        names.append(name)
    ax.hist(plot_data, color=colors, label=names,bins=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figname)

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

def plot_out(figname, batch, left_pred, left_title, right_pred, right_title, indices=None):
    left_actual = batch.y
    right_actual = batch.y
    num_anchors = torch.sum(batch.anchors)
    fig, (left, right) = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)
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

    if False:

        # visualize adjacency matrix
        num_nodes = len(batch.y)
        a = left
        pred = left_pred
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    if batch.adj[i][j]:
                        a.plot([batch.y[i,0],batch.y[j,0]], [batch.y[i,1],batch.y[j,1]], color="orange", alpha=0.2)
                        a.plot([pred[i,0],pred[j,0]], [pred[i,1],pred[j,1]], color="blue", alpha=0.2)

        # visualize neighbors
        if indices is not None:
            a = right
            pred = right_pred
            for i in range(num_nodes):
                for j in indices[i]:
                    a.plot([batch.y[i,0],batch.y[j,0]], [batch.y[i,1],batch.y[j,1]], color="orange", alpha=0.2)
                    a.plot([pred[i,0],pred[j,0]], [pred[i,1],pred[j,1]], color="blue", alpha=0.2)

    fig.tight_layout()
    fig.savefig(figname)
    print("Plot saved to",figname)

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
    threshold = 97
    nhid = 2000
    nout = 2
    dropout = 0.5
    lr = 0.01
    weight_decay = 0
    num_epochs = 200

    data_loader, num_nodes, noisy_distance_matrix = load_a_moment(eta=eta, Kref=Kref, threshold=threshold)
    print("dataset loaded...")

    # NOVEL PARAMS
    n_neighbors = 17 #8
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
    print("gcn RMSE:",gcn_error)
    print("...done")

    print("novel...")
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        num_anchors = anchor_locs.shape[0]
        start = time.time()
        X, Y, ff = separate_dataset_multiple_inits(noisy_distance_matrix, k0=k0, k1=k1, n_init=n_init, lam=lam, mu=mu, eps=eps)
        novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
        novel_solve_time = time.time()-start
        start = time.time()
        novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
        novel_error = torch.sqrt(novel_error).item()
        novel_predict_time = time.time()-start
        novel_total_time = novel_solve_time + novel_predict_time
        print("novel RMSE:",novel_error)
    print("...done")

    if False:
        print("solve direcct....")
        for batch in data_loader:
            anchor_locs = batch.y[batch.anchors]
            direct_pred = solve_direct(noisy_distance_matrix, anchor_locs, mode="Kabsch")
            direct_error = loss_fn(direct_pred[batch.nodes], batch.y[batch.nodes])
            direct_error = torch.sqrt(direct_error).item()
            print("direct RMSE:",direct_error)
        print("....done")

        print("reduce rank and solve direcct....")
        for batch in data_loader:
            denoised = denoise_via_SVD(noisy_distance_matrix**2,k=4,fill_diag=False,take_sqrt=False)
            anchor_locs = batch.y[batch.anchors]
            direct2_pred = solve_direct(denoised, anchor_locs, mode="Kabsch", dont_square=True)
            direct2_error = loss_fn(direct2_pred[batch.nodes], batch.y[batch.nodes])
            direct2_error = torch.sqrt(direct2_error).item()
            print("direct2 RMSE:",direct2_error)
        print("....done")

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

    # plot_out("_"+figname, batch, direct_pred, "Direct (Kabsch)", novel_pred, "Novel", indices=indices)
    plot_out(figname, batch, gcn_pred, "GCN (Yan et al.)", novel_pred,"Novel", indices=indices)

def experiment3():
    print("EXPERIMENT 3: CELLPHONE DATA")
    filename = "experiment3.txt"
    figname = "experiment3.jpg"
    num_anchors = 4
    loss_fn = torch.nn.MSELoss()

    # RSS TO DIST PARAMS
    eta = -2.9032366
    Kref = -46.120605

    # GCN PARAMS
    threshold = 5.0
    nhid = 2000
    nout = 2
    dropout = 0.5
    lr = 0.01
    weight_decay = 0
    num_epochs = 200

    data_loader, num_nodes, noisy_distance_matrix = load_cellphone_data(num_anchors=num_anchors, threshold=threshold)
    print("dataset loaded...")

    # NOVEL PARAMS
    n_neighbors = 5
    k0 = 4
    k1 = 0 #29
    lam = 1/(num_nodes**0.5)*0.7
    mu = 1/(num_nodes**0.5)*0.7
    eps = 0.001
    n_init = 100
    k1_init = 0 #num_nodes**2*(5/100)
    step_size = 0.85 #1

    if True:
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
        X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, plot=False)
        novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
        novel_solve_time = time.time()-start
        start = time.time()
        novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
        novel_error = torch.sqrt(novel_error).item()
        novel_predict_time = time.time()-start
        novel_total_time = novel_solve_time + novel_predict_time
        print("novel RMSE:",novel_error)
    print("...done")

    if False:
        print("solve direcct....")
        for batch in data_loader:
            anchor_locs = batch.y[batch.anchors]
            direct_pred = solve_direct(noisy_distance_matrix, anchor_locs, mode="Kabsch")
            direct_error = loss_fn(direct_pred[batch.nodes], batch.y[batch.nodes])
            direct_error = torch.sqrt(direct_error).item()
            print("direct RMSE:",direct_error)
        print("....done")

    if False:
        print("reduce rank and solve direcct....")
        for batch in data_loader:
            denoised = denoise_via_SVD(noisy_distance_matrix**2,k=4,fill_diag=False,take_sqrt=False)
            anchor_locs = batch.y[batch.anchors]
            direct2_pred = solve_direct(denoised, anchor_locs, mode="Kabsch", dont_square=True)
            direct2_error = loss_fn(direct2_pred[batch.nodes], batch.y[batch.nodes])
            direct2_error = torch.sqrt(direct2_error).item()
            print("direct2 RMSE:",direct2_error)
        print("....done")


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

        file_handle.write("RESULTS OF EXPERIMENT 3" + '\n')
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

    plot_out(figname, batch, gcn_pred, "GCN (Yan et al.)", novel_pred, "Novel", indices=indices)
    # plot_out("_"+figname, batch, direct_pred, "Direct (Kabsch)", novel_pred, "Novel")

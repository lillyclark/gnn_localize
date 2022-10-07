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
    n_neighbors = 15
    k0 = 4
    lam = 1/(num_nodes**0.5)*1.1
    mu = 1/(num_nodes**0.5)*1.1
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
        start = time.time()
        novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
        novel_error = torch.sqrt(novel_error).item()
        novel_predict_time = time.time()-start
        novel_total_time = novel_solve_time + novel_predict_time
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
        file_handle.write("Novel solve time:" + str(np.round(novel_solve_time,3)) + '\n')
        file_handle.write("Novel predict time:" + str(np.round(novel_predict_time,3)) + '\n')
        file_handle.write("Novel total time:" + str(np.round(novel_total_time,3)) + '\n')

        file_handle.close()
        print("Results written to", filename)

    write_()
    # plot_out(figname, batch, gcn_pred, "GCN (Yan et al.)", novel_pred, "Novel", indices=indices)

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

def experiment4():
    print("EXPERIMENT 4: vary n_neighbors")
    filename = "experiment4.txt"
    figname = "experiment4_runtime_ci.jpg"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # NOVEL PARAMS
    n_neighbor_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    # n_neighbor_list = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]
    # n_neighbor_list = [3,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,499]
    # n_neighbor_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
    # n_neighbor_list = [5,10,15,20,25,30,40,50,100,150,200,300]
    # n_neighbor_list = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51]
    # n_neighbor_list = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    k0 = 4
    lam = 1/(num_nodes**0.5)*1.1
    mu = 1/(num_nodes**0.5)*1.1
    eps = 0.001
    n_init = 1
    k1_init = num_nodes**2*(5/100)
    step_size = 1

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
        _ = 0
        result_list = []
        k1_list = []
        runtime_list = []
        res_ci = []
        run_ci = []
        for n_neighbors in n_neighbor_list:
            res = []
            run = []
            iters = 10
            for iter in range(iters):
                print("n_neighbors:",n_neighbors)
                _ += 1
                anchor_locs = batch.y[batch.anchors]
                noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
                start = time.time()
                X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, plot=False)
                novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
                novel_solve_time = time.time()-start
                start = time.time()
                novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
                novel_error = torch.sqrt(novel_error).item()
                novel_predict_time = time.time()-start
                novel_total_time = novel_solve_time + novel_predict_time
                res.append(novel_error)
                run.append(novel_total_time)
            res, run = np.array(res), np.array(run)
            writeLine(np.mean(res), _==len(n_neighbor_list))
            result_list.append(np.mean(res))
            runtime_list.append(np.mean(run))
            res_ci.append(1.96 * np.std(res)/np.sqrt(len(res)))
            run_ci.append(1.96 * np.std(run)/np.sqrt(len(run)))
            print("res:",res,"ci:",res_ci[-1])
            print("run:",run,"ci:",run_ci[-1])
    print("...done")
    writeLine(k1_list, True)

    fig, ax1 = plt.subplots(figsize=(6,3))

    ax1.plot(n_neighbor_list, result_list, marker='o',color='blue')
    ax1.fill_between(n_neighbor_list, (np.array(result_list)-np.array(res_ci)), (np.array(result_list)+np.array(res_ci)), color='blue', alpha=.1)
    ax1.set_xlabel('Number of neighbors')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(n_neighbor_list, runtime_list, marker='o', color='orange')
    ax2.fill_between(n_neighbor_list, (np.array(runtime_list)-np.array(run_ci)), (np.array(runtime_list)+np.array(run_ci)), color='orange', alpha=.1)
    ax2.set_ylabel('Runtime (solve + predict, sec)')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.tight_layout()
    plt.savefig(figname)
    print("plot saved to",figname)

def experiment5():
    print("EXPERIMENT 5: vary percentage nLOS")
    filename = "experiment5.txt"
    figname = "experiment5.jpg"
    num_nodes = 500
    num_anchors = 50
    loss_fn = torch.nn.MSELoss()

    # DATA PARAMS
    percent_nLOS_list = [0,10,20,30,40,50]
    # percent_nLOS_list = [0]

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
    lam = 1/(num_nodes**0.5)*1.1
    mu = 1/(num_nodes**0.5)*1.1
    eps = 0.001
    n_init = 1

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

        file_handle.write("percent_nLOS_list: " +str(percent_nLOS_list) + '\n')
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

    for p_nLOS in percent_nLOS_list:
        data_loader, num_nodes, noisy_distance_matrix = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS)
        print("dataset loaded...")

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
            X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
            novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
            novel_solve_time = time.time()-start
            start = time.time()
            novel_error = loss_fn(novel_pred[batch.nodes], batch.y[batch.nodes])
            novel_error = torch.sqrt(novel_error).item()
            novel_predict_time = time.time()-start
            novel_total_time = novel_solve_time + novel_predict_time
        print("...done")
        novel_list.append(novel_error)
        print("novel error:", novel_list)

    writeLine("Novel error:",False)
    writeLine(novel_list, True)
    writeLine("GCN error:",False)
    writeLine(gcn_list, True)

    fig, ax1 = plt.subplots(figsize=(6,3))
    ax1.plot(percent_nLOS_list, novel_list, marker='o',color='blue',label="Novel")
    ax1.plot(percent_nLOS_list, gcn_list, marker='o',color='orange',label="GCN")
    ax1.set_xlabel('% nLOS')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    plt.tight_layout()
    plt.savefig(figname)
    print("plot saved to",figname)

if __name__ == "__main__":
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    # _, _, _ = fake_dataset(500, 50, threshold=1.2, p_nLOS=10)
    # _, _, _ = their_dataset(500, 50, threshold=10000)

    # experiment1()
    # experiment2()
    # experiment3()
    # experiment4()
    experiment5()

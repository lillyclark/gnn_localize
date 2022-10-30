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

# DATA PARAMS
num_nodes = 500
num_anchors = 50
p_nLOS = 10
std = 0.1
noise_floor_dist = None

# GCN PARAMS
threshold = 1.2
nhid = 2000
nout = 2
dropout = 0.5
lr = 0.01
weight_decay = 0
num_epochs = 200

# NOVEL PARAMS
n_neighbors = 50
k0 = 4
lam = 0.01 #1/(num_nodes**0.5)
mu = 0.1 #1/(num_nodes**0.5)
eps = 0.001
n_init = 1
# k1_init = num_nodes**2*(5/100)
k1_init = 0
step_size = 1
eps_k1 = 0.01
constrain_solution=False

# COLORS
TRUE_COLOR = 'blue'
GCN_COLOR = 'orange'
SMILE_COLOR = 'deeppink'
ABLATION_COLOR = 'lightgreen'

def plot_rmse(figname, true_locs, to_eval):
    plot_data = []
    colors = []
    names = []
    color_dict = {"GCN":GCN_COLOR, "SMILE":SMILE_COLOR}
    fig, ax = plt.subplots(1,1,figsize=(6,3))
    for name, pred_locs in to_eval.items():
        rmse = np.linalg.norm(pred_locs.detach().numpy() - true_locs.detach().numpy(), axis=1)
        plot_data.append(rmse)
        colors.append(color_dict[name])
        names.append(name)
    ax.hist(plot_data, color=colors, label=names,bins=20)
    ax.legend()
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
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
    if 'GCN' not in left_title or 'SMILE' not in right_title:
        raise ValueError()
    left_actual = batch.y
    right_actual = batch.y
    num_anchors = torch.sum(batch.anchors)
    fig, (left, right) = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)
    left.scatter(left_actual[:num_anchors,0].detach().numpy(), left_actual[:num_anchors,1].detach().numpy(), label="true anchor", marker="x",color=TRUE_COLOR)
    left.scatter(left_pred[:num_anchors,0].detach().numpy(), left_pred[:num_anchors,1].detach().numpy(), label="pred anchor", marker="+",color=GCN_COLOR)
    left.scatter(left_actual[num_anchors:,0].detach().numpy(), left_actual[num_anchors:,1].detach().numpy(), label="true node",color=TRUE_COLOR)
    left.scatter(left_pred[num_anchors:,0].detach().numpy(), left_pred[num_anchors:,1].detach().numpy(), label="pred node",color=GCN_COLOR)
    left.set_title(left_title)
    # left.legend(ncol=2)
    right.scatter(right_actual[:num_anchors,0].detach().numpy(), right_actual[:num_anchors,1].detach().numpy(), label="true anchor", marker="x",color=TRUE_COLOR)
    right.scatter(right_pred[:num_anchors,0].detach().numpy(), right_pred[:num_anchors,1].detach().numpy(), label="pred anchor", marker="+",color=SMILE_COLOR)
    right.scatter(right_actual[num_anchors:,0].detach().numpy(), right_actual[num_anchors:,1].detach().numpy(), label="true node",color=TRUE_COLOR)
    right.scatter(right_pred[num_anchors:,0].detach().numpy(), right_pred[num_anchors:,1].detach().numpy(), label="pred node",color=SMILE_COLOR)
    right.set_title(right_title)
    # right.legend(ncol=2)
    # handles1, labels1 = right.get_legend_handles_labels()
    # handles2, labels2 = left.get_legend_handles_labels()
    # print(handles1, labels1)
    # handles, labels = set(handles1+handles2), set(labels1+labels2)
    # fig.legend(handles, labels, loc='lower center', ncol=4)

    if indices == "error_lines":
        print("trying to print error lines")
        num_nodes = len(batch.y)
        for i in range(num_nodes):
            left.plot([batch.y[i,0],left_pred[i,0]], [batch.y[i,1], left_pred[i,1]], color=GCN_COLOR, alpha=0.2)
            right.plot([batch.y[i,0],right_pred[i,0]], [batch.y[i,1], right_pred[i,1]], color=SMILE_COLOR, alpha=0.2)

    elif indices is not None:

        # visualize adjacency matrix
        num_nodes = len(batch.y)
        a = left
        pred = left_pred
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    if batch.adj[i][j]:
                        a.plot([batch.y[i,0],batch.y[j,0]], [batch.y[i,1],batch.y[j,1]], color=TRUE_COLOR, alpha=0.2)
                        a.plot([pred[i,0],pred[j,0]], [pred[i,1],pred[j,1]], color=GCN_COLOR, alpha=0.2)

        # visualize neighbors
        if indices is not None:
            a = right
            pred = right_pred
            for i in range(num_nodes):
                for j in indices[i]:
                    a.plot([batch.y[i,0],batch.y[j,0]], [batch.y[i,1],batch.y[j,1]], color=TRUE_COLOR, alpha=0.2)
                    a.plot([pred[i,0],pred[j,0]], [pred[i,1],pred[j,1]], color=SMILE_COLOR, alpha=0.2)

    fig.tight_layout()
    fig.savefig(figname)
    print("Plot saved to",figname)

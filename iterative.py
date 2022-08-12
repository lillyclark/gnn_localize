from process_dataset import *
from lle_like import solve_like_LLE, neighbors, barycenter_weights, weight_to_mat, solve_with_LRR, solve_iteratively
from decomposition import normalize_tensor, denoise_via_SVD, solve_direct
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds
from separate import *

if __name__=="__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    print("random seed is",0)

    num_nodes = 100
    num_anchors = 20
    print(num_nodes,"nodes",num_anchors,"num_anchors")
    true_locs, distance_matrix, k0, noise, nLOS, k1, measured = separable_dataset(num_nodes, num_anchors)
    print("k1:",k1)

    lam = 1/(num_nodes**0.5)
    mu = 1/(num_nodes**0.5)*0.1
    eps = 0.001
    n_init = 100
    print("lam:",lam)
    print("mu:",mu)
    print("eps:",eps)
    X, Y, ff = separate_dataset(measured, k0, k1, lam, mu)
    # print("n_init:",n_init)
    # X, Y, ff = separate_dataset_multiple_inits(measured, k0, k1, n_init=n_init, lam=lam, mu=mu, eps=eps)


    n_neighbors = 20
    print("n_neighbors:",n_neighbors)
    anchor_locs = true_locs[:num_anchors]

    pred_X = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True)

    pred_X_iterative = solve_iteratively(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True)

    loss_fn = torch.nn.MSELoss()

    loss_X = loss_fn(pred_X[num_anchors:], true_locs[num_anchors:])
    print(f"X LLE (RMSE):{torch.sqrt(loss_X).item()}")

    loss_it = loss_fn(pred_X_iterative[num_anchors:], true_locs[num_anchors:])
    print(f"X iterative (RMSE):{torch.sqrt(loss_it).item()}")

    fig2, axes2 = plt.subplots(1,2)

    ax = axes2[0]
    ax.scatter(pred_X[:num_anchors,0].detach().numpy(), pred_X[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
    ax.scatter(pred_X[num_anchors:,0].detach().numpy(), pred_X[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
    ax.legend()
    ax.set_title(f"solve LLE with X")

    ax = axes2[1]
    ax.scatter(pred_X_iterative[:num_anchors,0].detach().numpy(), pred_X_iterative[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
    ax.scatter(pred_X_iterative[num_anchors:,0].detach().numpy(), pred_X_iterative[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
    ax.legend()
    ax.set_title(f"solve iteratively with X")

    plt.show()

from process_dataset import *
from lle_like import solve_like_LLE, neighbors, barycenter_weights, weight_to_mat, solve_with_LRR
from decomposition import normalize_tensor, denoise_via_SVD, solve_direct
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds

def S_from_D(D, k1):
    D_ = D.flatten()
    S_ = torch.zeros_like(D_)
    s = torch.argsort(D_,descending=True)
    sk1 = s[:k1]
    S_[sk1] = 1
    return S_.reshape(D.shape)

def f(D, X, Y, lam, mu):
    A = torch.sum((D-X-Y)**2)
    B = lam*torch.sum(X**2)
    C = mu*torch.sum(Y**2)
    # print("f(D,X,Y)=",A+B+C)
    return A+B+C

def check_sparsity(X):
    count_pos = torch.sum(X>1e-9)
    return count_pos, X.shape[0]*X.shape[1]

def check_rank(X):
    return np.linalg.matrix_rank(X)

def reduce_rank(X,k=4):
    U, S, V = svds(X,k)
    X_ = U.dot(np.diag(S)).dot(V)
    return torch.Tensor(X_)

def solve_sparse_problem(D, X, mu, k1):
    D_ = D-X
    S = S_from_D(D_,k1)
    Y = S*D_/(1+mu)
    return Y

def solve_rank_problem(D, Y, lam, k0):
    D_ = D-Y
    return torch.Tensor(reduce_rank(D_/(1+lam), k0))
    # D_k0 = torch.Tensor(reduce_rank(D_,k0))
    # X = 1/(1+lam)*D_k0
    # return X

def separate_dataset(measured, k0, k1, lam=0.1, mu=0.1, eps=0.001, X=None, Y=None):
    D = measured**2
    if check_rank(D) == k0:
        print("already low rank")
        return D, torch.zeros_like(D), 0
    if X is None:
        X = torch.zeros_like(D)
    if Y is None:
        Y = torch.zeros_like(D)
    fi = f(D, X, Y, lam, mu)
    # print("fi:",fi)
    # fig, axes = plt.subplots(10,3)
    for iter in range(100):
        Y = solve_sparse_problem(D, X, mu, k1)
        X = solve_rank_problem(D, Y, lam, k0)
        ff = f(D, X, Y, lam, mu)
        if (fi-ff)/ff <= eps:
            print(iter,"ff:",ff)
            return X, Y, ff
        fi = ff
        # axes[iter][0].imshow(D)
        # axes[iter][1].imshow(X)
        # axes[iter][2].imshow(Y)
    # plt.show()
    return X, Y, ff

def separate_dataset_multiple_inits(measured, k0, k1, n_init=10, lam=0.1, mu=0.1, eps=0.001):
    best_X, best_Y, best_ff = separate_dataset(measured, k0, k1, lam, mu, eps)
    for init in range(1,n_init):
        init_X = reduce_rank(torch.rand(measured.shape)*torch.max(measured))
        X, Y, ff = separate_dataset(measured, k0, k1, lam, mu, eps, X=init_X)
        if ff < best_ff:
            best_X, best_Y, best_ff = X, Y, ff
    return best_X, best_Y, best_ff

def separate_dataset_find_k1(measured, k0, k1_init=0, step_size=1, n_init=1, lam=0.1, mu=0.1, eps=0.001, plot=False):
    """ step_size in percentage of edges """
    num_edges = int(measured.shape[0]*measured.shape[1])
    step_size_per = step_size/100
    step_size = int(num_edges*step_size/100)
    k1 = k1_init

    print("k1:",k1)
    X, Y, ff = separate_dataset_multiple_inits(measured, k0, k1, n_init=n_init, lam=lam, mu=mu, eps=eps)
    if ff == 0:
        return X, Y, ff

    while True:
        k1 += step_size
        if k1 > num_edges*(7/10):
            print("Estimated sparsity exceeded 70%")
            return X, Y, ff
        print("k1:",k1)
        X_, Y_, ff_ = separate_dataset_multiple_inits(measured, k0, int(k1), n_init=n_init, lam=lam, mu=mu, eps=eps)
        delta = (ff - ff_)/ff
        print("delta:",delta)
        if delta < step_size_per:
            print("***converged***")
            print("best guess is k1:", k1)
            return X_, Y_, ff_
        ff = ff_

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

    # print("CHECK RANK:")
    # print(check_rank(X))
    # print("CHECK SPARSITY:")
    # print(check_sparsity(Y))

    # fig, axes = plt.subplots(2,3)
    # axes[0][0].imshow(distance_matrix)
    # axes[0][0].set_title("original distance matrix")
    # axes[0][1].imshow(distance_matrix**2)
    # axes[0][1].set_title("squared distance matrix")
    # axes[0][2].imshow(X)
    # axes[0][2].set_title("low rank matrix")
    #
    # axes[1][0].imshow(nLOS)
    # axes[1][0].set_title("true sparse noise")
    # axes[1][1].imshow(nLOS**2)
    # axes[1][1].set_title("squared sparse noise")
    # axes[1][2].imshow(Y)
    # axes[1][2].set_title("sparse matrix")

    n_neighbors = 10
    print("n_neighbors:",n_neighbors)
    anchor_locs = true_locs[:num_anchors]

    pred_direct_PCA = solve_direct(measured, anchor_locs, mode="Kabsch")
    pred_orig = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, measured, dont_square=False)

    denoised = denoise_via_SVD(measured**2,k=4,fill_diag=False,take_sqrt=False)
    pred_denoised = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, denoised, dont_square=True)
    pred_denoised_direct = solve_direct(denoised, anchor_locs, mode="Kabsch")

    pred_X = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True)
    pred_X_direct = solve_direct(X, anchor_locs, mode="Kabsch")

    loss_fn = torch.nn.MSELoss()

    loss_direct_PCA = loss_fn(pred_direct_PCA[num_anchors:], true_locs[num_anchors:])
    print(f"direct (RMSE):{torch.sqrt(loss_direct_PCA).item()}")

    loss_orig = loss_fn(pred_orig[num_anchors:], true_locs[num_anchors:])
    print(f"orig (RMSE):{torch.sqrt(loss_orig).item()}")

    loss_denoised = loss_fn(pred_denoised[num_anchors:], true_locs[num_anchors:])
    print(f"denoised (RMSE):{torch.sqrt(loss_denoised).item()}")

    loss_denoised_direct = loss_fn(pred_denoised_direct[num_anchors:], true_locs[num_anchors:])
    print(f"denoised direct (RMSE):{torch.sqrt(loss_denoised_direct).item()}")

    loss_X = loss_fn(pred_X[num_anchors:], true_locs[num_anchors:])
    print(f"X (RMSE):{torch.sqrt(loss_X).item()}")

    loss_X_direct = loss_fn(pred_X_direct[num_anchors:], true_locs[num_anchors:])
    print(f"X direct (RMSE):{torch.sqrt(loss_X_direct).item()}")

    fig2, axes2 = plt.subplots(2,3)

    ax = axes2[0][0]
    ax.scatter(pred_direct_PCA[:num_anchors,0].detach().numpy(), pred_direct_PCA[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
    ax.scatter(pred_direct_PCA[num_anchors:,0].detach().numpy(), pred_direct_PCA[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
    ax.legend()
    ax.set_title(f"solve directly (Kabsch) D")

    ax = axes2[1][0]
    ax.scatter(pred_orig[:num_anchors,0].detach().numpy(), pred_orig[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
    ax.scatter(pred_orig[num_anchors:,0].detach().numpy(), pred_orig[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
    ax.legend()
    ax.set_title(f"solve LLE D")

    ax = axes2[1][1]
    ax.scatter(pred_denoised[:num_anchors,0].detach().numpy(), pred_denoised[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
    ax.scatter(pred_denoised[num_anchors:,0].detach().numpy(), pred_denoised[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
    ax.legend()
    ax.set_title(f"solve LLE rank_reduce(D)")

    ax = axes2[0][1]
    ax.scatter(pred_denoised_direct[:num_anchors,0].detach().numpy(), pred_denoised_direct[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
    ax.scatter(pred_denoised_direct[num_anchors:,0].detach().numpy(), pred_denoised_direct[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
    ax.legend()
    ax.set_title(f"solve directly (Kabsch) rank_reduce(D)")

    ax = axes2[1][2]
    ax.scatter(pred_X[:num_anchors,0].detach().numpy(), pred_X[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
    ax.scatter(pred_X[num_anchors:,0].detach().numpy(), pred_X[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
    ax.legend()
    ax.set_title(f"solve LLE with X")

    ax = axes2[0][2]
    ax.scatter(pred_X_direct[:num_anchors,0].detach().numpy(), pred_X_direct[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[:num_anchors,0].detach().numpy(), true_locs[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
    ax.scatter(pred_X_direct[num_anchors:,0].detach().numpy(), pred_X_direct[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
    ax.scatter(true_locs[num_anchors:,0].detach().numpy(), true_locs[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
    ax.legend()
    ax.set_title(f"solve directly (Kabsch) X")

    plt.show()

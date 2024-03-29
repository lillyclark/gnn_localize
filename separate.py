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
    A = torch.linalg.norm(D-X-Y, ord="fro")**2
    # A = torch.sum((D-X-Y)**2)
    B = lam*torch.linalg.norm(X)**2
    # B = lam*torch.sum(X**2)
    C = mu*torch.linalg.norm(Y)**2
    # C = mu*torch.sum(Y**2)
    # print("f(D,X,Y)=",A+B+C)
    return A+B+C

def check_sparsity(X):
    count_pos = torch.sum(X>1e-9)
    return count_pos, X.shape[0]*X.shape[1]

def check_rank(X):
    return np.linalg.matrix_rank(X)

def reduce_rank(X,k=4):
    if X.shape[0] == k:
        U, S, V = np.linalg.svd(X,k)
    else:
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

def constrain_X(X):
    X[X<0] = 0
    return (X+X.T)/2

def constrain_Y(Y):
    return (Y+Y.T)/2

def separate_dataset(measured, k0, k1, lam=0.1, mu=0.1, eps=0.001, X=None, Y=None, constrain_solution=False):
    D = measured**2
    if X is None:
        X = torch.zeros_like(D)
    if Y is None:
        Y = torch.zeros_like(D)
    fi = f(D, X, Y, lam, mu)
    for iter in range(100):
        Y = solve_sparse_problem(D, X, mu, k1)
        X = solve_rank_problem(D, Y, lam, k0)
        if constrain_solution:
            X = constrain_X(X)
            Y = constrain_Y(Y)
        ff = f(D, X, Y, lam, mu)
        if (fi-ff)/fi <= eps:
            return X, Y, ff
        fi = ff
    return X, Y, ff

def separate_dataset_multiple_inits(measured, k0, k1, n_init=10, lam=0.1, mu=0.1, eps=0.001, constrain_solution=False):
    best_X, best_Y, best_ff = separate_dataset(measured, k0, k1, lam=lam, mu=mu, eps=eps, constrain_solution=constrain_solution)
    for init in range(1,n_init):
        init_X = reduce_rank(torch.rand(measured.shape)*torch.max(measured))
        X, Y, ff = separate_dataset(measured, k0, k1, lam=lam, mu=mu, eps=eps, X=init_X, constrain_solution=constrain_solution)
        if ff < best_ff:
            best_X, best_Y, best_ff = X, Y, ff
    return best_X, best_Y, best_ff

def separate_dataset_find_k1(measured, k0, k1_init=0, step_size=1, n_init=1, lam=0.1, mu=0.1, eps=0.001, eps_k1=0.01, plot=False, constrain_solution=False):
    if check_rank(measured**2) == k0:
        print("already low rank")
        return measured**2, torch.zeros_like(measured), 0, 0

    num_edges = int(measured.shape[0]*measured.shape[1])
    step_size = int(num_edges*step_size/100)
    k1 = k1_init
    X, Y, fi = separate_dataset_multiple_inits(measured, k0, k1, n_init=n_init, lam=lam, mu=mu, eps=eps, constrain_solution=constrain_solution)
    for iter in range(100):
        k1 += step_size
        X, Y, ff = separate_dataset_multiple_inits(measured, k0, int(k1), n_init=n_init, lam=lam, mu=mu, eps=eps, constrain_solution=constrain_solution)
        if (fi-ff)/fi <= eps_k1:
            return X, Y, ff, k1
        fi = ff
    return X, Y, ff, k1

if __name__=="__main__":
    torch.manual_seed(0)
    D = torch.rand((100,100))*5
    X = torch.rand((100,100))*2
    Y = torch.rand((100,100))*3
    res = f(D, X, Y, 0.1, 0.01)
    print(res)

import torch.optim as optim
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh
from process_dataset import matrix_from_locs

np.random.seed(1)
torch.manual_seed(1)

num_nodes = 4
num_anchors = 3
threshold = 10

true_locs = torch.Tensor([[0,0],[0,2],[2,0],[2,2]])
distance_matrix = matrix_from_locs(true_locs)
noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
noise.fill_diagonal_(0)
noisy_distance_matrix = distance_matrix + noise
noisy_distance_matrix = (noisy_distance_matrix.T+noisy_distance_matrix)/2

adj = (noisy_distance_matrix<threshold)
adj.fill_diagonal_(0)
print(adj)
indices = np.where(adj.numpy())[1].reshape(adj.shape[0],-1)
print(indices)

def barycenter_weights1(X, Y, indices, reg=1e-5):
    n_samples, n_neighbors = indices.shape
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)
    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B

# res = barycenter_weights1(true_locs.numpy(), true_locs.numpy(), indices)
# print(np.round(res,2))

def barycenter_weights(distance_matrix, indices, reg=1e-5):
    n_samples, n_neighbors = indices.shape
    B = np.empty((n_samples, n_neighbors))
    v = np.ones(n_neighbors)
    D = distance_matrix.numpy()
    for i, ind in enumerate(indices):
        C = np.empty((n_neighbors, n_neighbors))
        for j in range(n_neighbors):
            for k in range(n_neighbors):
                C[j][k] = (D[i][ind[k]]**2 + D[ind[j]][i]**2 - D[ind[j]][ind[k]]**2)/2
        trace = np.trace(C)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        C.flat[:: n_neighbors + 1] += R
        w = solve(C, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B

res = barycenter_weights(noisy_distance_matrix, indices, reg=1e-3)
print(np.round(res,2))

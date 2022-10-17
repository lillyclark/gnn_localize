import torch.optim as optim
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric
from scipy.linalg import eigh, svd, qr, solve, lstsq
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh
from process_dataset import matrix_from_locs
from decomposition import normalize_tensor, reduce_rank, denoise_via_SVD

def solve_with_LRR(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=False):
    indices = neighbors(noisy_distance_matrix, n_neighbors)
    start = time.time()
    res = reduce_and_find_weights(noisy_distance_matrix, indices, reg=1e-3,dont_square=dont_square)
    print(f"{time.time()-start} to reduce and find weight mat")
    mat = weight_to_mat(res, indices)
    I_minus_W = np.eye(num_nodes)-mat
    RHS = I_minus_W[:,:num_anchors]
    RHS = RHS.dot(anchor_locs)
    LHS = -1*I_minus_W[:,num_anchors:]
    start = time.time()
    node_locs, res, rnk, s = lstsq(LHS, RHS)
    print("RES:",res)
    print(f"{time.time()-start} to find locs")
    pred = np.vstack((anchor_locs,node_locs))
    return pred

def reduce_and_find_weights(distance_matrix, indices, reg=1e-5, dont_square=False):
    n_samples, n_neighbors = indices.shape
    B = np.empty((n_samples, n_neighbors))
    v = np.ones(n_neighbors)
    D = distance_matrix.numpy()
    for i, ind in enumerate(indices):
        idx = [i]+ind.tolist()
        idx = np.ix_(idx,idx)
        D_ = D[idx]
        D_ = reduce_rank(D_,k=4)
        C = np.empty((n_neighbors, n_neighbors))
        for j in range(n_neighbors):
            for k in range(n_neighbors):
                if dont_square:
                    C[j][k] = (D_[0][k+1] + D_[j+1][0] - D_[j+1][k+1])/2
                else:
                    C[j][k] = (D_[0][k+1]**2 + D_[j+1][0]**2 - D_[j+1][k+1]**2)/2
        trace = np.trace(C)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        C.flat[:: n_neighbors + 1] += R
        try:
            w = solve(C, v, assume_a='pos')
        except np.linalg.LinAlgError:
            # print('in barycenter_weights, matrix C is singular -> use least squares')
            w, res, rnk, s = lstsq(C, v)
        B[i, :] = w / np.sum(w)
    return B

def barycenter_weights(distance_matrix, indices, reg=1e-5, dont_square=False):
    n_samples, n_neighbors = indices.shape
    B = np.empty((n_samples, n_neighbors))
    v = np.ones(n_neighbors)
    D = distance_matrix.numpy()

    if not dont_square:
        D = D**2

    for i, ind in enumerate(indices):
        row = D[i,ind]
        m1 = np.outer(np.ones(n_neighbors), row)
        col = D[ind,i]
        m2 = np.outer(col, np.ones(n_neighbors))
        C = (m1+m2-D[ind][:,ind])/2
        # print(C)

        # C = np.empty((n_neighbors, n_neighbors))
        # for j in range(n_neighbors):
        #     for k in range(n_neighbors):
        #         C[j][k] = (D[i][ind[k]] + D[ind[j]][i] - D[ind[j]][ind[k]])/2
        # print(C)

        trace = np.trace(C)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        C.flat[:: n_neighbors + 1] += R
        try:
            w = solve(C, v, assume_a='pos')
        except np.linalg.LinAlgError:
            # print('in barycenter_weights, matrix C is singular -> use least squares')
            perfect = False
            w, res, rnk, s = lstsq(C, v)
        B[i, :] = w / np.sum(w)
    return B

def weight_to_mat(weights, indices):
    n_samples, n_neighbors = indices.shape
    mat = np.zeros((n_samples, n_samples))
    for i, ind in enumerate(indices):
        mat[i][ind] = weights[i]
    return mat

def neighbors(distance_matrix, n_neighbors):
    indices = np.argsort(distance_matrix.numpy(), axis=1)
    return indices[:,1:n_neighbors+1]

def solve_like_LLE(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=False,anchors_as_neighbors=False, return_indices=False):
    if anchors_as_neighbors:
        indices = np.vstack([np.linspace(0,n_neighbors-1,n_neighbors,dtype=int)]*num_nodes)
    else:
        indices = neighbors(noisy_distance_matrix, n_neighbors)
    start = time.time()
    res = barycenter_weights(noisy_distance_matrix, indices, reg=1e-3,dont_square=dont_square)
    # print(f"{time.time()-start} to find weight mat")
    mat = weight_to_mat(res, indices)
    I_minus_W = np.eye(num_nodes)-mat
    RHS = I_minus_W[:,:num_anchors]
    RHS = RHS.dot(anchor_locs)
    LHS = -1*I_minus_W[:,num_anchors:]
    start = time.time()
    node_locs, res, rnk, s = lstsq(LHS, RHS)
    # print("RES:",res)
    # print(f"{time.time()-start} to find locs")
    pred = np.vstack((anchor_locs,node_locs))
    if return_indices:
        return torch.Tensor(pred), indices
    return torch.Tensor(pred)

def solve_iteratively(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=False):
    indices = neighbors(noisy_distance_matrix, n_neighbors)
    # start = time.time()
    res = barycenter_weights(noisy_distance_matrix, indices, reg=1e-3,dont_square=dont_square)
    # print(f"{time.time()-start} to find weight mat")
    mat = weight_to_mat(res, indices)

    final_locs = np.zeros((num_nodes,2))
    final_locs[:num_anchors] = anchor_locs
    known = np.zeros(num_nodes, dtype=bool)
    known[:num_anchors] = True

    known_before = np.sum(known)
    learned_something = True
    K = n_neighbors

    while K >= 0:

        while learned_something:

            W_known = mat[:,known]
            W_unknown = mat[:,np.logical_not(known)]
            counts = np.count_nonzero(W_known, axis=1)
            solve_these = counts>=K
            solve_these[known] = False
            if np.sum(solve_these)==0:
                break
            print("considering",np.sum(solve_these),"rows...")

            if K == n_neighbors:
                final_locs[solve_these] = W_known[solve_these].dot(final_locs[known])

            else:
                LHS = (mat-np.eye(num_nodes))[:,solve_these]
                RHS = (np.eye(num_nodes)-mat)[:,known].dot(final_locs[known])
                final_locs[solve_these], _, _, _ = lstsq(LHS,RHS)

            known[solve_these] = True
            known_after = np.sum(known)
            if known_after == num_nodes:
                print("done!!!!")
                return torch.Tensor(final_locs)
            if known_after == known_before:
                learned_something = False
            known_before = known_after

        print("stopped learning with K=",K)
        K -= 1
        learned_something = True

    print("uh oh, couldn't solve everything?")
    return torch.Tensor(final_locs)



    for row in W_known:
        if np.count_nonzero(row) == n_neighbors:
            print("this node's neighbors are all anchors!")
            print(row)
            loc = row.dot(anchor_locs)
            print("this node's loc is:")
            print(loc)
            return

    I_minus_W = np.eye(num_nodes)-mat
    RHS = I_minus_W[:,:num_anchors]
    RHS = RHS.dot(anchor_locs)
    LHS = -1*I_minus_W[:,num_anchors:]
    start = time.time()
    node_locs, res, rnk, s = lstsq(LHS, RHS)
    # print("RES:",res)
    # print(f"{time.time()-start} to find locs")
    pred = np.vstack((anchor_locs,node_locs))
    return torch.Tensor(pred)

if __name__=="__main__":
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    # weights = torch.rand((20,5))
    # indices = torch.choice(np.arange(20),(20,5))
    # m = weight_to_mat(weights, indices)

    num_nodes = 500
    n_neighbors = 50
    distance_matrix = torch.rand((num_nodes, num_nodes))*5
    distance_matrix = (distance_matrix + distance_matrix.T)/2
    distance_matrix.fill_diagonal_(0)
    distance_matrix = torch.round(distance_matrix)
    indices = neighbors(distance_matrix, n_neighbors)
    start = time.time()
    for i in range(10):
        W = barycenter_weights(distance_matrix, indices)
    print(time.time()-start)
    print(W)

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


# def barycenter_weights1(X, Y, indices, reg=1e-5):
#     n_samples, n_neighbors = indices.shape
#     B = np.empty((n_samples, n_neighbors))
#     v = np.ones(n_neighbors)
#     for i, ind in enumerate(indices):
#         A = Y[ind]
#         C = A - X[i]  # broadcasting
#         G = np.dot(C, C.T)
#         trace = np.trace(G)
#         if trace > 0:
#             R = reg * trace
#         else:
#             R = reg
#         G.flat[:: n_neighbors + 1] += R
#         w = solve(G, v, sym_pos=True)
#         B[i, :] = w / np.sum(w)
#     return B

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
    perfect = True
    for i, ind in enumerate(indices):
        C = np.empty((n_neighbors, n_neighbors))
        for j in range(n_neighbors):
            for k in range(n_neighbors):
                if dont_square:
                    C[j][k] = (D[i][ind[k]] + D[ind[j]][i] - D[ind[j]][ind[k]])/2
                else:
                    C[j][k] = (D[i][ind[k]]**2 + D[ind[j]][i]**2 - D[ind[j]][ind[k]]**2)/2
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
    # if perfect:
    #     print("able to recover weights exactly")
    # else:
    #     print("using least squares to recover weights")
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

def test_this():
    np.random.seed(1)
    torch.manual_seed(1)

    num_nodes = 4
    num_anchors = 3

    true_locs = torch.Tensor([[0,0],[0,2],[2,0],[2,2]])
    distance_matrix = matrix_from_locs(true_locs)
    noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
    noise.fill_diagonal_(0)
    noisy_distance_matrix = distance_matrix + noise
    noisy_distance_matrix = (noisy_distance_matrix.T+noisy_distance_matrix)/2

    indices = neighbors(distance_matrix, 3)
    print(indices)

    res = barycenter_weights(noisy_distance_matrix, indices, reg=1e-5)
    # res = barycenter_weights(distance_matrix, indices, reg=1e-5)
    print(np.round(res,2))

    print("sparse weights")
    mat = weight_to_mat(res, indices)
    print(mat)

    pred = np.dot(mat, true_locs)

    c = ["red","orange","green","blue"]
    for n in [0,1,2,3]:
        plt.scatter(pred[n,0], pred[n,1], label=str(n), color=c[n])
        plt.scatter(true_locs[n,0].detach().numpy(), true_locs[n,1].detach().numpy(), label=str(n)+"true", color=c[n], marker="x")
    plt.legend()
    plt.title('four nodes demo')
    plt.show()

def test_solve_4():
    np.random.seed(1)
    torch.manual_seed(1)

    num_nodes = 4
    num_anchors = 3

    true_locs = torch.Tensor([[0,0],[0,2],[2,0],[2,2]])
    distance_matrix = matrix_from_locs(true_locs)
    noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
    noise.fill_diagonal_(0)
    noisy_distance_matrix = distance_matrix + noise
    noisy_distance_matrix = (noisy_distance_matrix.T+noisy_distance_matrix)/2

    indices = neighbors(noisy_distance_matrix, 3)
    print(indices)

    res = barycenter_weights(noisy_distance_matrix, indices, reg=1e-5)
    print(np.round(res,2))

    print("sparse weights")
    mat = weight_to_mat(res, indices)
    print(mat)

    I_minus_W = np.eye(num_nodes)-mat
    RHS = I_minus_W[:,:num_anchors]
    RHS = RHS.dot(true_locs[:num_anchors])

    LHS = -1*I_minus_W[:,num_anchors:]

    node_locs, res, rnk, s = lstsq(LHS, RHS)

    pred = np.vstack((true_locs[:num_anchors],node_locs))

    c = ["red","orange","green","blue"]
    for n in [0,1,2,3]:
        plt.scatter(pred[n,0], pred[n,1], label=str(n), color=c[n])
        plt.scatter(true_locs[n,0].detach().numpy(), true_locs[n,1].detach().numpy(), label=str(n)+"true", color=c[n], marker="x")
    plt.legend()
    plt.title('four nodes demo')
    plt.show()

def test_solve_10():
    np.random.seed(1)
    torch.manual_seed(1)

    num_nodes = 10
    num_anchors = 3
    n_neighbors = 9

    true_locs = torch.Tensor([[0,0],[0,2],[2,0],[2,2],[1,1],[0,1],[1,0],[1,2],[2,1],[-1,-1]])
    distance_matrix = matrix_from_locs(true_locs)
    noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
    noise.fill_diagonal_(0)
    noisy_distance_matrix = distance_matrix + noise
    noisy_distance_matrix = (noisy_distance_matrix.T+noisy_distance_matrix)/2

    euclidean_matrix = noisy_distance_matrix**2
    noisy_distance_matrix = denoise_via_SVD(euclidean_matrix,k=4,fill_diag=False,take_sqrt=False)
    pred = solve_like_LLE(num_nodes, num_anchors, n_neighbors, true_locs[:num_anchors], noisy_distance_matrix, dont_square=True)

    c = ["red","orange","yellow","green","blue","pink","purple","cyan","magenta","grey"]
    for n in range(num_nodes):
        plt.scatter(pred[n,0], pred[n,1], label=str(n), color=c[n])
        plt.scatter(true_locs[n,0].detach().numpy(), true_locs[n,1].detach().numpy(), label=str(n)+" true", color=c[n], marker="x")
    plt.legend()
    plt.title('solve like LLE')
    plt.show()

# test_solve_10()

from process_dataset import *
from models import *
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
from scipy.linalg import eigh, svd, qr, solve, lstsq

def normalize_tensor(x):
    D = x.sum(1)
    r_inv = (D**-0.5).flatten()
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(torch.mm(r_mat_inv,x),r_mat_inv)
    return mx

def reduce_rank(X,k=4,make_sym=False):
    if make_sym:
        X=(X+X.T)/2
    U, S, V = svds(X,k)
    X_ = U.dot(np.diag(S)).dot(V)
    return X_

def denoise_via_SVD(euclidean_matrix,k=4,fill_diag=False,take_sqrt=False):
    x = euclidean_matrix
    new_x = torch.Tensor(reduce_rank(x,k))
    if fill_diag:
        new_x.fill_diagonal_(0)
    if take_sqrt:
        new_x = new_x**0.5
    return new_x

def denoise_via_adj(noisy_distance_matrix,K,threshold=1.2,normalize=False):
    print("original noisy rank:",np.linalg.matrix_rank(noisy_distance_matrix))
    adjacency_matrix = (noisy_distance_matrix<threshold).float()
    if normalize:
        adjacency_matrix = normalize_tensor(adjacency_matrix)
    adjK = np.linalg.matrix_power(adjacency_matrix,K)
    thresholded_noisy_distance_matrix  = noisy_distance_matrix.clone()
    thresholded_noisy_distance_matrix[thresholded_noisy_distance_matrix>threshold] = 0.0
    new_x = thresholded_noisy_distance_matrix
    for i in range(K):
        if normalize:
            new_x = normalize_tensor(torch.mm(adjacency_matrix,new_x))
        else:
            new_x = torch.mm(adjacency_matrix,new_x)
    print("new rank:",np.linalg.matrix_rank(new_x))
    return new_x, adjK, thresholded_noisy_distance_matrix

def solve_direct(noisy_distance_matrix, anchor_locs, mode="None", dont_square=False):
    x = noisy_distance_matrix
    num_nodes = x.shape[0]
    num_anchors = anchor_locs.shape[0]
    M = np.zeros(x.shape)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if dont_square:
                M[i][j] = (x[0][j] + x[i][0] - x[i][j])/2
            else:
                M[i][j] = (x[0][j]**2 + x[i][0]**2 - x[i][j]**2)/2

    q, v = np.linalg.eig(M)
    print("rank of M:",np.linalg.matrix_rank(M, tol=0.001))
    # print("q:",q)
    # plt.plot(q.real)
    # plt.show()

    locs = np.zeros((num_nodes,2))
    # locs[:,0] = np.sqrt(q[0]).real*v[:,0].real
    # locs[:,1] = np.sqrt(q[1]).real*v[:,1].real

    locs[:,0] = np.sqrt(q[0])*v[:,0]
    locs[:,1] = np.sqrt(q[1])*v[:,1]


    # print(anchor_locs)
    # print(locs)

    if mode == "None":
        pass

    if mode == "Kabsch":
        A = anchor_locs.numpy()
        B = locs[:num_anchors]
        n, m = A.shape
        EA = np.mean(A, axis=0)
        EB = np.mean(B, axis=0)
        VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)
        H = ((A - EA).T @ (B - EB)) / n
        U, D, VT = np.linalg.svd(H)

        # TRY d = 1
        d = 1
        S_pos = np.diag([1] * (m - 1) + [d])
        c_pos = VarA / np.trace(np.diag(D) @ S_pos)

        # TRY d = 1
        d = -1
        S_neg = np.diag([1] * (m - 1) + [d])
        c_neg = VarA / np.trace(np.diag(D) @ S_neg)

        if abs(c_pos - 1) < abs(c_neg - 1):
            S, c = S_pos, c_pos
        else:
            S, c = S_neg, c_neg

        R = U @ S @ VT
        t = EA - c * R @ EB
        locs = np.array([t + c * R @ b for b in locs])

    return torch.Tensor(locs)

if __name__=="__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    num_nodes = 10
    num_anchors = 2
    threshold = 1
    data_loader, num_nodes, noisy_distance_matrix = fake_dataset(num_nodes, num_anchors, threshold=threshold)

    loss_fn = torch.nn.MSELoss()

    for batch in data_loader:
        # euclidean_matrix = noisy_distance_matrix**2
        # y = denoise_via_SVD(euclidean_matrix,k=4,fill_diag=False,take_sqrt=False)
        # print(y)
        # print(np.diagonal(y))
        # fig, (ax1, ax2) = plt.subplots(1,2)
        # ax1.imshow(euclidean_matrix)
        # ax1.set_title('noisy euclidean matrix '+str(np.linalg.matrix_rank(euclidean_matrix)))
        # ax2.imshow(y)
        # ax2.set_title('de-noised '+str(np.linalg.matrix_rank(y)))
        # plt.show()

        # euclidean_matrix = noisy_distance_matrix**2
        # y = denoise_via_SVD(euclidean_matrix,k=4,fill_diag=True,take_sqrt=True)
        # fig, (ax1, ax2) = plt.subplots(1,2)
        # ax1.imshow(noisy_distance_matrix)
        # ax1.set_title('noisy dist matrix '+str(np.linalg.matrix_rank(noisy_distance_matrix)))
        # ax2.imshow(y)
        # ax2.set_title('de-noised '+str(np.linalg.matrix_rank(y)))
        # plt.show()

        # upto = 11
        # # fig, axes = plt.subplots(1,upto+1)
        # fig, axes = plt.subplots(2,6)
        # axes[0][0].imshow(noisy_distance_matrix)
        # axes[0][0].set_title('noisy dist matrix '+str(np.linalg.matrix_rank(noisy_distance_matrix)))
        #
        # for i in range(1,upto+1):
        #     j,k = int(i/6), i%6
        #     print(i,j,k)
        #     y = denoise_via_adj(noisy_distance_matrix,K=i-1,threshold=1.2)
        #     axes[j][k].imshow(y)
        #     if i == 1:
        #         axes[j][k].set_title('thresholded '+str(np.linalg.matrix_rank(y)))
        #     else:
        #         axes[j][k].set_title('de-noised '+str(np.linalg.matrix_rank(y)))
        # plt.show()


        # fig, axes = plt.subplots(2,6)
        # axes[0][0].imshow(noisy_distance_matrix)
        # axes[0][0].set_title('noisy dist matrix '+str(np.linalg.matrix_rank(noisy_distance_matrix)))
        # y, adj1 = denoise_via_adj(noisy_distance_matrix,K=1)
        # axes[1][0].imshow(adj1)
        # axes[1][0].set_title('adj matrix '+str(np.linalg.matrix_rank(adj1)))
        #
        # for i in range(1,6):
        #     y, adj = denoise_via_adj(noisy_distance_matrix,K=i,threshold=1.2)
        #     axes[0][i].imshow(y)
        #     axes[0][i].set_title('K='+str(i)+' de-noised '+str(np.linalg.matrix_rank(y)))
        #     axes[1][i].imshow(adj)
        #     axes[1][i].set_title('adj^'+str(i)+' '+str(np.linalg.matrix_rank(adj)))
        # plt.show()

        fig, axes = plt.subplots(1,6)
        K=100
        axes[0].imshow(noisy_distance_matrix)
        axes[0].set_title('Dist')
        y, adj, dist_thr = denoise_via_adj(noisy_distance_matrix,K=1,normalize=True)
        axes[1].imshow(dist_thr)
        axes[1].set_title('Dist_thr')
        axes[2].imshow(adj)
        axes[2].set_title('Adj')
        yK, adjK, _ = denoise_via_adj(noisy_distance_matrix,K=K,normalize=True)
        axes[3].imshow(adjK)
        axes[3].set_title('Adj^'+str(K))
        axes[4].imshow(yK)
        axes[4].set_title('Adj^'+str(K)+' @ Dist_thr')
        axes[5].imshow(1/(1+noisy_distance_matrix))
        axes[5].set_title('Similarity (1/(1+Dist))')
        print(np.round(yK.numpy(),2))
        plt.show()

        # fig, axes = plt.subplots(1,4)
        # axes[0].imshow(adjK)
        # axes[0].set_title('Adj^'+str(K))
        # axes[1].imshow(yK)
        # axes[1].set_title('Adj^'+str(K)+' @ Dist_thr')
        # lambda_dist_thr, u_dist_thr = np.linalg.eig(dist_thr)
        # axes[2].imshow(u_dist_thr)
        # axes[2].set_title('Eigenvalues of Dist_thr')
        lambda_adj, u_adj = np.linalg.eig(adj)
        print(np.round(lambda_adj,2))
        print(np.round(u_adj,2))
        # axes[3].imshow(u_adj)
        # axes[3].set_title('Eigenvalues of Adj')
        # plt.show()


    # for batch in data_loader:
    #     x = batch.x.to_dense().numpy()
    #     x = (x+x.T)/2
    #     print(np.round(x,2))
    #
    #     M = np.zeros(x.shape)
    #
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             M[i][j] = (x[0][j]**2 + x[i][0]**2 - x[i][j]**2)/2
    #
    #     q, v = np.linalg.eig(M)
    #     locs = np.zeros((num_nodes,2))
    #     locs[:,0] = np.sqrt(q[0])*v[:,0]
    #     locs[:,1] = np.sqrt(q[1])*v[:,1]
    #
    #     pca = PCA(n_components=2)
    #     anchor_mean = torch.mean(batch.y[batch.anchors], axis=0)
    #     print(anchor_mean)
    #     anchor_pca = torch.Tensor(pca.fit_transform(batch.y[batch.anchors]))
    #     anchor_M = pca.components_
    #
    #     locs = pca.fit_transform(locs)
    #     locs = np.matmul(locs, anchor_M)
    #     locs = torch.Tensor(locs + anchor_mean.numpy())
    #
    #
    #     pred = torch.Tensor(locs)
    #     loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    #     loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
    #     print(f"train:{loss_train.item()}, val:{loss_val.item()}")
    #
    # plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
    # plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual")
    # plt.legend()
    # plt.title('decomposition')
    # plt.show()

# d = np.array([[0, 2, 2, 2*2**0.5],[2,0,2*2**0.5,2],[2,2*2**0.5,0,2],[2*2**0.5,2,2,0]])

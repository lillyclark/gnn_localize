import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from torch_geometric.data import Data
import torch_geometric
from torch_geometric.loader import DataLoader
import scipy.sparse as sp

from scipy.io import loadmat
from sklearn.linear_model import LinearRegression

# from torch.nn.functional import normalize

pdist = torch.nn.PairwiseDistance(p=2)

def normalize(x, use_sparse=True):
    D = np.array(x.sum(1))
    r_inv = np.power(D, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    if use_sparse:
        r_mat_inv = sp.diags(r_inv)
    else:
        r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(x).dot(r_mat_inv)
    return mx

def normalize_tensor(x):
    D = x.sum(1)
    r_inv = (D**-0.5).flatten()
    r_inv[torch.isnan(r_inv)]=0
    r_inv[torch.isinf(r_inv)]=0
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(torch.mm(r_mat_inv,x),r_mat_inv)
    return mx

def matrix_from_locs(locs):
    num_nodes = locs.shape[0]
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = pdist(locs[i].unsqueeze(0), locs[j].unsqueeze(0))
            distance_matrix[i][j] = d
    return distance_matrix

def their_dataset(num_nodes, num_anchor, threshold=1.0):
    if threshold is None:
        threshold = 10000
    # m = loadmat("./GNN-For-localization/Networks/8anchor_1000agent_0PercentNLOS_smallLOS.mat")
    m = loadmat("./GNN-For-localization/Networks/8anchor_1000agent_10PercentNLOS_mediumLOS.mat")
    Range_Mat = m["Range_Mat"][108:num_nodes+108,108:num_nodes+108]
    Dist_Mat = m["Dist_Mat"][108:num_nodes+108,108:num_nodes+108]
    labels = m["nodes"][108:num_nodes+108]
    Range = Dist_Mat.copy()
    Range[Range > threshold] = 0
    Range_tem = Range.copy()
    Range_tem[Range_tem > 0] = 1

    # features = sp.csr_matrix(Range, dtype=np.float64)
    # Adj = sp.csr_matrix(Range_tem, dtype=np.float64)
    features = Range
    Adj = Range_tem

    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))+1e-9
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    # features = normalize(features)
    # adj = normalize(Adj + sp.eye(Adj.shape[0]))
    features = normalize(features)
    adj = normalize(Adj + np.eye(Adj.shape[0]))


    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # features = sparse_mx_to_torch_sparse_tensor(features)
    # true_locs = torch.FloatTensor(labels)
    # normalized_adjacency_matrix = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.Tensor(features)
    true_locs = torch.FloatTensor(labels)
    normalized_adjacency_matrix = torch.Tensor(adj)

    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchor):
        anchor_mask[a] = True
    for n in range(num_anchor,num_nodes):
        node_mask[n] = True
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data]), num_nodes, Range_Mat

def fake_dataset(num_nodes, num_anchors, threshold=1.0, p_nLOS=10, std=0.1, nLOS_max=10, noise_floor_dist=None):
    # nodes is total nodes, including anchors
    true_locs = torch.rand((num_nodes,2))*5
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    nLOS_noise = torch.zeros((num_nodes, num_nodes))
    p_nLOS = p_nLOS/100

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i < j:
                d = pdist(true_locs[i].unsqueeze(0), true_locs[j].unsqueeze(0))
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d

                if np.random.random() < p_nLOS:
                    uniform_noise = torch.rand(())*nLOS_max
                    nLOS_noise[i][j] = uniform_noise
                    nLOS_noise[j][i] = uniform_noise

    noise = np.random.normal(loc=0.0,scale=std,size=(num_nodes,num_nodes))
    noise = torch.Tensor(noise)
    noise.fill_diagonal_(0)

    # p_nLOS = p_nLOS/100
    # nLOS = np.random.choice([0, 1], size=(num_nodes,num_nodes), p=[1-p_nLOS, p_nLOS])
    # nLOS = torch.Tensor(nLOS)
    # nLOS.fill_diagonal_(0)
    # nLOS_noise = torch.rand((num_nodes,num_nodes))*nLOS_max

    true_k1 = np.count_nonzero(nLOS_noise.numpy())
    # print(true_k1/(num_nodes*(num_nodes-1)))

    noisy_distance_matrix = distance_matrix + noise + nLOS_noise

    if noise_floor_dist:
        print("distances over", noise_floor_dist, "are measured as", np.ceil(5*2**0.5))
        # turn distances above a threshold into noise floor distances
        extra_k1 = np.count_nonzero(noisy_distance_matrix>noise_floor_dist)
        noisy_distance_matrix[noisy_distance_matrix>noise_floor_dist] = np.ceil(5*2**0.5)
        print("original k1:", true_k1, "new k1:", true_k1+extra_k1)
        true_k1 += extra_k1

    adjacency_matrix = (noisy_distance_matrix<threshold).float()
    thresholded_noisy_distance_matrix  = noisy_distance_matrix.clone()
    thresholded_noisy_distance_matrix[thresholded_noisy_distance_matrix>threshold] = 0.0

    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))+1e-9
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return torch.Tensor(mx)

    features = normalize(thresholded_noisy_distance_matrix)
    normalized_adjacency_matrix = normalize(adjacency_matrix)

    # features = normalize_tensor(thresholded_noisy_distance_matrix)
    # normalized_adjacency_matrix = normalize_tensor(adjacency_matrix)

    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchors):
        anchor_mask[a] = True
    for n in range(num_anchors,num_nodes):
        node_mask[n] = True
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data]), num_nodes, noisy_distance_matrix, true_k1

def is_anchor(str_name):
    if str_name == 'scom-base1':
        return True
    if '-' in str_name:
        return False
    else:
        return True

def is_not_anchor(str_name):
    return not is_anchor(str_name)

def pathloss_to_dist(pathloss, eta, Kref):
    return 10**((pathloss-Kref)/(eta*10))

def get_eta_Kref(pathlosses, distances):
    # print(pathlosses.shape)
    # print(distances.shape)
    distances = 10*np.log10(distances)
    reg = LinearRegression().fit(distances.reshape(-1,1), pathlosses)
    return reg.coef_, reg.intercept_

def pick_a_moment(filename='datasets/sep18d_clean.csv'):
    start = time.time()
    data = pd.read_csv(filename,header=0)
    data_ = data[data['tags'].isna()] # only real data
    print(f"Loaded dataset in {time.time()-start} seconds")

    times = data_['timestamp']
    unique_times = tuple(set(times))
    print("There are",len(unique_times),"moments to choose from")

    max_nodes = 0
    best_i = None
    best_moment = None

    for i, moment in enumerate(unique_times):
        data = data_[data_['timestamp']==moment]
        transmitters = set(data['transmitter'])
        receivers = set(data['receiver'])
        nodes = tuple(transmitters.union(receivers))
        node_idx = dict(zip(nodes, range(len(nodes))))
        num_nodes = len(nodes)
        if num_nodes > max_nodes:
            max_nodes = num_nodes
            best_i = i
            best_moment = moment

    print("Best i:", best_i)
    print("Best moment:", best_moment)
    print("Max nodes:", max_nodes)
    return best_moment, max_nodes


def load_a_moment(filename='datasets/sep18d_clean.csv', moment=1165, eta=3.2, Kref=40, threshold=50.0):
    start = time.time()
    data = pd.read_csv(filename,header=0)
    data = data[data['tags'].isna()] # only real data
    print(f"Loaded dataset in {time.time()-start} seconds")

    times = data['timestamp']
    unique_times = tuple(set(times))
    # print("There are",len(unique_times),"moments to choose from")
    moment = unique_times[moment]
    # print("moment:",moment)

    data = data[data['timestamp']==moment]
    transmitters = set(data['transmitter'])
    receivers = set(data['receiver'])
    nodes = tuple(transmitters.union(receivers))
    nodes = sorted(nodes)
    nodes = sorted(nodes,key=is_not_anchor)
    # print(nodes)
    node_idx = dict(zip(nodes, range(len(nodes))))
    num_nodes = len(nodes)

    # print(data)

    noisy_distance_matrix = np.zeros((num_nodes, num_nodes))
    true_locs = np.zeros((num_nodes, 2)) # TODO 3D
    anchor_mask = torch.Tensor([is_anchor(node) for node in nodes]).bool()
    node_mask = ~anchor_mask

    pathlosses = []
    distances = []

    LOS = torch.zeros((num_nodes, num_nodes))
    nLOS = torch.zeros((num_nodes, num_nodes))

    for j in range(len(data)):
        tx, rx = data.iloc[j]['transmitter'], data.iloc[j]['receiver']
        tx_id, rx_id = node_idx[tx], node_idx[rx]
        pathloss = float(data.iloc[j]['measured_path_loss_dB'])
        noisy_distance_matrix[tx_id][rx_id] = pathloss_to_dist(pathloss, eta, Kref)
        true_locs[tx_id] = data.iloc[j]['transmitter_x'], data.iloc[j]['transmitter_y']
        true_locs[rx_id] = data.iloc[j]['receiver_x'], data.iloc[j]['receiver_y']
        # pathlosses.append(pathloss)
        # distances.append(np.sum((true_locs[tx_id]-true_locs[rx_id])**2)**0.5)
        if data.iloc[j]['visible'] == 0.0:
            nLOS[tx_id][rx_id] = 1
        if data.iloc[j]['visible'] == 1.0:
            LOS[tx_id][rx_id] = 1
            pathlosses.append(pathloss)
            distances.append(np.sum((true_locs[tx_id]-true_locs[rx_id])**2)**0.5)

    print("*****")
    k1_nLOS = torch.sum(nLOS)
    print("nLOS:",k1_nLOS)

    print("eta, Kref:")
    print(get_eta_Kref(np.array(pathlosses), np.array(distances)))
    # assert False

    k1_missing = np.sum(noisy_distance_matrix==0)-num_nodes
    print("missing:", k1_missing)

    print("k1:",k1_nLOS+k1_missing)
    print("%:",(k1_nLOS+k1_missing)/(num_nodes*(num_nodes-1)))
    print("*****")

    noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
    # print("NOISY MATRIX")
    # print(np.round(noisy_distance_matrix.numpy(),0))
    true_locs = torch.Tensor(true_locs)
    # print("TRUE LOCATIONS")
    # print(true_locs)

    # print("TRUE DIST MATRIX")
    true_dist = matrix_from_locs(true_locs)
    # print(np.round(true_dist))

    # fig, (ax0,ax1,ax2) = plt.subplots(1,3)
    #
    # ax0.hist(noisy_distance_matrix[LOS!=0].flatten()-true_dist[LOS!=0].flatten(),bins=20)
    # ax0.set_title("how much noise is there for LOS links")
    LOS_err = (noisy_distance_matrix[LOS!=0].flatten()-true_dist[LOS!=0].flatten()).numpy()
    print("LOS mean & std dev",np.mean(LOS_err),np.std(LOS_err))
    #
    # # ax1.hist(noisy_distance_matrix[nLOS!=0].flatten()-true_dist[nLOS!=0].flatten(),bins=20)
    # # ax1.set_title("how much noise is there for nLOS")
    nLOS_err = (noisy_distance_matrix[nLOS!=0].flatten()-true_dist[nLOS!=0].flatten()).numpy()
    print("nLOS mean & std dev",np.mean(nLOS_err),np.std(nLOS_err))
    #
    # ax1.hist(noisy_distance_matrix[noisy_distance_matrix!=0].flatten()-true_dist[noisy_distance_matrix!=0].flatten(),bins=20)
    # ax1.set_title("how much noise is there when a measurement is present")
    meas_err = (noisy_distance_matrix[noisy_distance_matrix!=0].flatten()-true_dist[noisy_distance_matrix!=0].flatten()).numpy()
    print("mean mean & std dev",np.mean(meas_err),np.std(meas_err))
    #
    print("Assuming we know the actual max distance....")
    fill_max = torch.max(true_dist)+1
    print("FILL MAX:", fill_max)
    noisy_distance_matrix[noisy_distance_matrix==0] = fill_max
    # np.fill_diagonal(noisy_distance_matrix,0)
    noisy_distance_matrix.fill_diagonal_(0)
    #
    # ax2.hist(noisy_distance_matrix.flatten()-true_dist.flatten(),bins=20)
    # ax2.set_title("how much noise is there when missing="+str(fill_max))
    all_err = (noisy_distance_matrix.flatten()-true_dist.flatten()).numpy()
    print("all mean & std dev",np.mean(all_err),np.std(all_err))

    # plt.show()

    features = noisy_distance_matrix.clone()
    adjacency_matrix = features < threshold
    features[features > threshold]=0
    print("TODO: ROW NORMALIZE INSTEAD")
    features = normalize_tensor(features)
    normalized_adjacency_matrix = normalize_tensor(adjacency_matrix.float())
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data],shuffle=False), num_nodes, noisy_distance_matrix

def load_cellphone_data(num_anchors=3, threshold=10.0):
    num_nodes = 11

    anchor_mask = torch.Tensor([i < num_anchors for i in range(num_nodes)]).bool()
    node_mask = ~anchor_mask

    true_locs = torch.Tensor([[1.0, 14.0, 3.0, 21.0, 9.0, 22.0, 10.0, 3.0, 27.0, 20.0, 18.0],
    	     [7.0, 3.0, 19.0, 18.0, 33.0, 3.0, 11.0, 32.0, 27.0, 12.0, 34.0]]).T

    # convert to meters
    true_locs *= 0.3048

    # First Row represents the x-coordinates of the nodes, and the second row represents the y-coordinates. All values are in feet.

    pathlosses = np.array([[-100.00, -68.308, -62.299, -67.051, -68.141, -66.400, -60.864, -77.891, -68.217, -69.553, -68.678],
    	      [-67.414, -100.00, -68.205, -65.299, -69.623, -55.270, -59.656, -71.891, -69.178, -65.439, -74.507],
    	      [-60.025, -67.713, -100.00, -65.146, -64.812, -67.982, -58.420, -62.949, -70.291, -68.334, -69.354],
    	      [-68.820, -67.432, -67.988, -100.00, -68.111, -64.830, -65.680, -70.307, -56.207, -53.938, -67.278],
    	      [-66.090, -68.082, -64.801, -64.279, -100.00, -73.314, -63.975, -49.990, -62.357, -71.328, -54.147],
    	      [-67.572, -55.854, -70.355, -64.174, -77.715, -100.00, -77.591, -88.658, -68.432, -57.822, -73.303],
    	      [-62.527, -62.622, -62.680, -65.855, -68.123, -70.461, -100.00, -70.402, -73.127, -65.211, -79.361],
    	      [-77.521, -74.398, -65.662, -70.068, -50.838, -86.564, -70.062, -100.00, -70.560, -81.057, -67.722],
    	      [-68.896, -70.385, -72.194, -54.641, -63.342, -67.637, -70.941, -69.543, -100.00, -65.686, -63.720],
    	      [-66.488, -63.775, -66.978, -51.330, -69.295, -56.240, -63.301, -78.221, -65.041, -100.00, -70.042],
    	      [-72.078, -80.838, -73.341, -68.369, -57.963, -76.385, -79.549, -68.691, -66.980, -73.718, -100.00]])

    true_dist = matrix_from_locs(true_locs)
    # print("true dist squared:")
    # print(true_dist**2)

    # pathlosses = 10*np.log10(np.array(true_dist))
    # print("making matrix symmetric right from the start")
    # pathlosses = (pathlosses + pathlosses.T)/2

    pl = []
    dist = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                pl.append(pathlosses[i][j])
                dist.append(true_dist[i][j])

    eta, Kref = get_eta_Kref(np.array(pl), np.array(dist))
    print(eta, Kref)

    # eta, Kref = -3, -50

    # logdist = 10*np.log10(np.array(dist))
    # pl = np.array(pl)
    # plt.scatter(logdist, pl)
    # print(min(logdist), max(logdist))
    # plt.plot([min(logdist), max(logdist)],
    #         [eta.item()*min(logdist)+Kref.item(), eta.item()*max(logdist)+Kref.item()])
    # plt.show()

    if False:
        d_ = pathloss_to_dist(pl, eta, Kref)
        noise = d_ - dist
        plt.plot(noise)
        plt.show()

    noisy_distance_matrix = pathloss_to_dist(pathlosses, eta, Kref)
    noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
    noisy_distance_matrix.fill_diagonal_(0)

    all_err = (noisy_distance_matrix.flatten()-true_dist.flatten()).numpy()
    print("all mean & std dev",np.mean(all_err),np.std(all_err))

    features = noisy_distance_matrix.clone()
    adjacency_matrix = features < threshold
    features[features > threshold]=0
    print("TODO: ROW NORMALIZE INSTEAD")
    features = normalize_tensor(features)
    normalized_adjacency_matrix = normalize_tensor(adjacency_matrix.float())
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data],shuffle=False), num_nodes, noisy_distance_matrix


if __name__=="__main__":
    print("executing process_dataset.py")
    data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(500, 50, threshold=1.2, p_nLOS=10)

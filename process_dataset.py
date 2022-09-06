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

def modified_adj(num_nodes, num_anchors, threshold=1.0):
    # nodes is total nodes, including anchors
    true_locs = torch.rand((num_nodes,2))*5
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = pdist(true_locs[i].unsqueeze(0), true_locs[j].unsqueeze(0))
            distance_matrix[i][j] = d
    noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
    noise.fill_diagonal_(0)
    # make symmetric (TODO)
    noise = (noise + noise.T)/2
    noisy_distance_matrix = distance_matrix + noise

    adjacency_matrix = 1/((noisy_distance_matrix+1)**3)
    print("adj:")
    print(adjacency_matrix)
    thresholded_noisy_distance_matrix  = noisy_distance_matrix.clone()
    thresholded_noisy_distance_matrix[thresholded_noisy_distance_matrix>threshold] = 0
    features = normalize(thresholded_noisy_distance_matrix, p=1.0, dim=1)
    normalized_adjacency_matrix = normalize(adjacency_matrix, p=1.0, dim=1)

    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchors):
        anchor_mask[a] = True
    for n in range(num_anchors,num_nodes):
        node_mask[n] = True
    # edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(normalized_adjacency_matrix)
    # data = Data(x=features.to_sparse(), edge_index=edge_index, edge_attr=edge_attr, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    # data = Data(x=features.to_sparse(), adj=normalized_adjacency_matrix.to_sparse(), y=true_locs, anchors=anchor_mask, nodes=node_mask)
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data]), num_nodes, noisy_distance_matrix

def their_dataset(num_nodes, num_anchor, threshold=1.0):
    if threshold is None:
        threshold = 10000
    # m = loadmat("./GNN-For-localization/Networks/8anchor_1000agent_0PercentNLOS_smallLOS.mat")
    m = loadmat("./GNN-For-localization/Networks/8anchor_1000agent_10PercentNLOS_mediumLOS.mat")
    Range_Mat = m["Range_Mat"][:num_nodes,:num_nodes]
    Dist_Mat = m["Dist_Mat"][:num_nodes,:num_nodes]
    labels = m["nodes"][:num_nodes]
    Range = Range_Mat.copy()
    Range[Range > threshold] = 0
    Range_tem = Range.copy()
    Range_tem[Range_tem > 0] = 1

    features = sp.csr_matrix(Range, dtype=np.float64)
    Adj = sp.csr_matrix(Range_tem, dtype=np.float64)

    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    features = normalize(features)
    adj = normalize(Adj + sp.eye(Adj.shape[0]))

    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    features = sparse_mx_to_torch_sparse_tensor(features)
    true_locs = torch.FloatTensor(labels)
    normalized_adjacency_matrix = sparse_mx_to_torch_sparse_tensor(adj)

    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchor):
        anchor_mask[a] = True
    for n in range(num_anchor,num_nodes):
        node_mask[n] = True
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data]), num_nodes, Range_Mat

def separable_dataset(num_nodes, num_anchors):
    true_locs = torch.rand((num_nodes,2))*5
    # true_locs[0] = torch.Tensor([0,0])
    # true_locs[1] = torch.Tensor([0,1])
    # true_locs[2] = torch.Tensor([1,0])
    k0 = 4
    distance_matrix = torch.Tensor(matrix_from_locs(true_locs))
    # noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
    noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
    print("zero-mean noise variance is ",0.04)
    noise.fill_diagonal_(0)
    p_nLOS = 1/10
    print("prob of nLOS is",p_nLOS)
    nLOS = np.random.choice([0, 1], size=(num_nodes,num_nodes), p=[1-p_nLOS, p_nLOS])
    nLOS = torch.Tensor(nLOS)
    nLOS.fill_diagonal_(0)
    k1 = int(torch.sum(nLOS).item())
    print("nLOS noise is U[0,10]")
    nLOS_noise = torch.rand((num_nodes,num_nodes))*10
    # print("nLOS noise is 10s")
    # nLOS_noise = torch.ones((num_nodes,num_nodes))*10
    nLOS = nLOS*nLOS_noise
    measured = distance_matrix+noise+nLOS
    return true_locs, distance_matrix, k0, noise, nLOS, k1, measured

def fake_dataset(num_nodes, num_anchors, threshold=1.0):
    # nodes is total nodes, including anchors
    true_locs = torch.rand((num_nodes,2))*5
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = pdist(true_locs[i].unsqueeze(0), true_locs[j].unsqueeze(0))
            distance_matrix[i][j] = d
    noise = torch.randn((num_nodes,num_nodes))*(0.00**0.5)
    noise.fill_diagonal_(0)

    p_nLOS = 1/10
    print("prob of nLOS is",p_nLOS)
    nLOS = np.random.choice([0, 1], size=(num_nodes,num_nodes), p=[1-p_nLOS, p_nLOS])
    nLOS = torch.Tensor(nLOS)
    nLOS.fill_diagonal_(0)
    print("nLOS noise is U[0,10]")
    nLOS_noise = torch.rand((num_nodes,num_nodes))*10
    print("how many nLOS measurements?")
    print(np.count_nonzero(nLOS.numpy()))

    noisy_distance_matrix = distance_matrix + noise + (nLOS*nLOS_noise)

    if threshold is not None:
        # turn distances above a threshold into noise floor distances
        noise_floor_distance_matrix  = noisy_distance_matrix.clone()
        print("how many distance measurements are above the threshold?")
        c = torch.sum(noise_floor_distance_matrix>threshold).item()
        print(c,"ie", c/(num_nodes**2)*100, "%")
        # plt.axvline(c,color="black",label="true missing")
        noise_floor_distance_matrix[noise_floor_distance_matrix>threshold] = np.ceil(5*2**0.5)

        adjacency_matrix = (noisy_distance_matrix<threshold).float()
        thresholded_noisy_distance_matrix  = noisy_distance_matrix.clone()
        thresholded_noisy_distance_matrix[thresholded_noisy_distance_matrix>threshold] = 0.0

    else:
        noise_floor_distance_matrix = noisy_distance_matrix
        adjacency_matrix = torch.ones_like(noisy_distance_matrix).float()
        thresholded_noisy_distance_matrix  = noisy_distance_matrix

    features = normalize_tensor(thresholded_noisy_distance_matrix)
    normalized_adjacency_matrix = normalize_tensor(adjacency_matrix)

    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchors):
        anchor_mask[a] = True
    for n in range(num_anchors,num_nodes):
        node_mask[n] = True
    # edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(normalized_adjacency_matrix)
    # data = Data(x=features.to_sparse(), edge_index=edge_index, edge_attr=edge_attr, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    # data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    # return DataLoader([data]), num_nodes, noisy_distance_matrix
    return DataLoader([data]), num_nodes, noise_floor_distance_matrix

def nLOS_dataset(num_nodes, num_anchors, threshold=1.0):
    true_locs = torch.rand((num_nodes,2))*5
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = pdist(true_locs[i].unsqueeze(0), true_locs[j].unsqueeze(0))
            distance_matrix[i][j] = d
    noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
    noise.fill_diagonal_(0)
    for i in range(int(num_nodes/10)):
        noise[i][i+1] += 0.05

    for j in range(int(num_nodes/10)+1,2*int(num_nodes/10)):
        noise[j][j+1] += 5

    noisy_distance_matrix = distance_matrix + noise

    adjacency_matrix = (noisy_distance_matrix<threshold).float()
    thresholded_noisy_distance_matrix  = noisy_distance_matrix.clone()
    thresholded_noisy_distance_matrix[thresholded_noisy_distance_matrix>threshold] = 0.0
    features = normalize_tensor(thresholded_noisy_distance_matrix)
    normalized_adjacency_matrix = normalize_tensor(adjacency_matrix)

    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchors):
        anchor_mask[a] = True
    for n in range(num_anchors,num_nodes):
        node_mask[n] = True
    # edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(normalized_adjacency_matrix)
    # data = Data(x=features.to_sparse(), edge_index=edge_index, edge_attr=edge_attr, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    # data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data]), num_nodes, noisy_distance_matrix

def no_noise_dataset(num_nodes, num_anchors, threshold=1.0):
    # nodes is total nodes, including anchors
    true_locs = torch.rand((num_nodes,2))*5
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = pdist(true_locs[i].unsqueeze(0), true_locs[j].unsqueeze(0))
            distance_matrix[i][j] = d

    adjacency_matrix = (distance_matrix<threshold).float()
    features = distance_matrix

    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchors):
        anchor_mask[a] = True
    for n in range(num_anchors,num_nodes):
        node_mask[n] = True
    # edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(normalized_adjacency_matrix)
    # data = Data(x=features.to_sparse(), edge_index=edge_index, edge_attr=edge_attr, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    data = Data(x=features.to_sparse(), adj=adjacency_matrix.to_sparse(), y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data]), num_nodes

def scoped_dataset(num_nodes, num_anchors, threshold=3, anchor_locs=None):
    true_locs = torch.rand((num_nodes,2))*5
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = pdist(true_locs[i].unsqueeze(0), true_locs[j].unsqueeze(0))
            distance_matrix[i][j] = d
    noise = torch.randn((num_nodes,num_nodes))*(0.04**0.5)
    noise.fill_diagonal_(0)
    noisy_distance_matrix = distance_matrix + noise

    ref_points_per_node = threshold
    # features = torch.zeros((num_nodes, ref_points_per_node))
    features = torch.zeros((num_nodes, num_nodes))
    adjacency_matrix = torch.eye(num_nodes)
    for i in range(num_nodes):
        ind = np.argpartition(distance_matrix.numpy()[i][:num_anchors], ref_points_per_node)[:ref_points_per_node]
        for ref, ref_point in enumerate(ind):
            # features[i][ref] = distance_matrix[i][ref_point]
            # features[i][ref+1] = ref_point
            features[i][ref_point] = distance_matrix[i][ref_point]
            adjacency_matrix[i][ref_point] = 1.0

    features = normalize(features, p=1.0, dim=1)
    normalized_adjacency_matrix = normalize(adjacency_matrix, p=1.0, dim=1)
    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchors):
        anchor_mask[a] = True
    for n in range(num_anchors,num_nodes):
        node_mask[n] = True
    # edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(normalized_adjacency_matrix)
    # data = Data(x=features.to_sparse(), edge_index=edge_index, edge_attr=edge_attr, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    data = Data(x=features.to_sparse(), adj=normalized_adjacency_matrix.to_sparse(), y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data]), num_nodes


def process_dataset(filename, batch_size, threshold=1000, fake_links=False):

    start = time.time()
    # data = pd.read_csv('datasets/sep18d_clean.csv',header=0)
    data = pd.read_csv(filename,header=0)
    if not fake_links:
        data = data[data['tags'].isna()] # only real data
    print(f"Loaded dataset in {time.time()-start} seconds")

    times = data['timestamp']
    unique_times = tuple(set(times))

    transmitters = set(data['transmitter'])
    receivers = set(data['receiver'])
    nodes = tuple(transmitters.union(receivers))

    # debug
    # nodes = ('scom-husky2','scom-spot4','scom-spot3','scom-husky3','scom-husky1','scom12','scom-base1','scom-spot1')

    node_index = {}
    for node in nodes:
        node_index[node] = len(node_index)
    num_nodes = len(nodes)

    graphs = []

    for i in [168]:
    # for i in range(len(unique_times)):
        print(i)

        graph = data[data['timestamp']==unique_times[i]]

        distance_matrix = np.zeros((num_nodes, num_nodes))
        labels = np.zeros((num_nodes, 3))
        anchor_mask = torch.zeros(num_nodes).bool()
        node_mask = torch.zeros(num_nodes).bool()

        edge_index_u = []
        edge_index_v = []
        edge_attr = []

        for j in range(len(graph)):
            tx, rx = graph.iloc[j]['transmitter'], graph.iloc[j]['receiver']
            pathloss = float(graph.iloc[j]['measured_path_loss_dB'])
            if pathloss < threshold:
                edge_index_u.append(node_index[tx])
                edge_index_v.append(node_index[rx])
                # edge_attr.append(pathloss)
                edge_attr.append(1)
                distance_matrix[node_index[tx]][node_index[rx]] = pathloss
                # debug
                x = torch.Tensor([graph.iloc[j]['transmitter_x'], graph.iloc[j]['transmitter_y'], graph.iloc[j]['transmitter_z']])
                y = torch.Tensor([graph.iloc[j]['receiver_x'], graph.iloc[j]['receiver_y'], graph.iloc[j]['receiver_z']])
                # distance_matrix[node_index[tx]][node_index[rx]] = pdist(x.unsqueeze(0), y.unsqueeze(0))

            labels[node_index[tx]] = graph.iloc[j]['transmitter_x'], graph.iloc[j]['transmitter_y'], graph.iloc[j]['transmitter_z']
            labels[node_index[rx]] = graph.iloc[j]['receiver_x'], graph.iloc[j]['receiver_y'], graph.iloc[j]['receiver_z']

            for n in [tx, rx]:
                if is_anchor(n):
                    anchor_mask[node_index[n]] = True
                else:
                    node_mask[node_index[n]] = True

        distance_matrix = torch.Tensor(distance_matrix)
        print("average links")
        labels = torch.Tensor(labels)
        edge_index = torch.Tensor([edge_index_u,edge_index_v]).long()
        edge_attr = torch.Tensor(edge_attr)

        # normalize
        print("distance matrix:")
        print(distance_matrix)
        distance_matrix = normalize(distance_matrix, p=1.0, dim=1)
        print("after normalize:")
        print(distance_matrix)


        adj = torch_geometric.utils.to_dense_adj(edge_index, edge_attr=edge_attr, max_num_nodes=num_nodes)
        print("is the adjacency_matrix symmetric?")
        print(torch.all(adj.transpose(0, 1) == adj))
        print("adj:")
        print(adj)
        adj = normalize(adj+torch.eye(num_nodes), p=1.0, dim=1)
        print("after normalize:")
        print(adj)
        edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)

        # print(edge_attr)

        graph_data = Data(x=distance_matrix.to_sparse(), adj=adj, edge_index=edge_index, edge_attr=edge_attr, y=labels, anchors=anchor_mask, nodes=node_mask)
        graphs.append(graph_data)

    data_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    return data_loader, nodes

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
    # print("Assuming we know the actual max distance....")
    fill_max = torch.max(true_dist)+1
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

    pl = []
    dist = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                pl.append(pathlosses[i][j])
                dist.append(true_dist[i][j])

    eta, Kref = get_eta_Kref(np.array(pl), np.array(dist))
    print(eta, Kref)

    # logdist = 10*np.log10(np.array(dist))
    # pl = np.array(pl)
    # plt.scatter(logdist, pl)
    # print(min(logdist), max(logdist))
    # plt.plot([min(logdist), max(logdist)],
    #         [eta.item()*min(logdist)+Kref.item(), eta.item()*max(logdist)+Kref.item()])
    # plt.show()

    noisy_distance_matrix = pathloss_to_dist(pathlosses, eta, Kref)
    noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
    noisy_distance_matrix.fill_diagonal_(0)

    all_err = (noisy_distance_matrix.flatten()-true_dist.flatten()).numpy()
    print("all mean & std dev",np.mean(all_err),np.std(all_err))

    features = noisy_distance_matrix.clone()
    adjacency_matrix = features < threshold
    features[features > threshold]=0
    features = normalize_tensor(features)
    normalized_adjacency_matrix = normalize_tensor(adjacency_matrix.float())
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data],shuffle=False), num_nodes, noisy_distance_matrix


if __name__=="__main__":
    print("executing process_dataset.py")
    # load_a_moment(eta=4.57435973, Kref=13.111276899657597)
    # pick_a_moment()
    load_cellphone_data()

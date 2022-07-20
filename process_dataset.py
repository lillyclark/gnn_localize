import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from torch_geometric.data import Data
import torch_geometric
from torch_geometric.loader import DataLoader

from torch.nn.functional import normalize

pdist = torch.nn.PairwiseDistance(p=2)

def matrix_from_locs(locs):
    num_nodes = locs.shape[0]
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = pdist(locs[i].unsqueeze(0), locs[j].unsqueeze(0))
            distance_matrix[i][j] = d
    return distance_matrix

def is_anchor(str_name):
    if 'spot4' in str_name:
        return False
    else:
        return True

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

def fake_dataset(num_nodes, num_anchors, threshold=1.0):
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

    adjacency_matrix = (noisy_distance_matrix<threshold).float()
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
    data = Data(x=features.to_sparse(), adj=normalized_adjacency_matrix.to_sparse(), y=true_locs, anchors=anchor_mask, nodes=node_mask)
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

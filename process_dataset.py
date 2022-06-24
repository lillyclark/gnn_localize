import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def process_dataset(filename, batch_size):

    start = time.time()
    # data = pd.read_csv('datasets/sep18d_clean.csv',header=0)
    data = pd.read_csv(filename,header=0)
    data = data[data['tags'].isna()] # only real data
    print(f"Loaded dataset in {time.time()-start} seconds")

    # Loaded dataset in 2.125419855117798 seconds
    # Index(['Unnamed: 0', 'timestamp_min', 'distance_int', 'timestamp', 'receiver',
    #        'transmitter', 'receiver_x', 'receiver_y', 'receiver_z',
    #        'transmitter_x', 'transmitter_y', 'transmitter_z', 'freq_MHz',
    #        'txpw_actual_dBm', 'receiver_pose_time', 'transmitter_pose_time',
    #        'visible', 'distance', 'maybe_occupied_space', 'occupied_space',
    #        'free_space', 'unknown_space', 'two_ray_G_dB', 'diffraction_G_dB',
    #        'noise_level_dBm', 'received_power_dBm', 'measured_path_loss_dB',
    #        'snr_dB', 'theoretical_udp_throughput_Mbps',
    #        'estimated_udp_throughput_Mbps', 'loss_rate_percent',
    #        'total_air_time_percent', 'total_data_rate_Mbps', 'num_spatial_streams',
    #        'strict_visible', 'strict_non_visible', 'logdist', 'not_free_space',
    #        'tags', 'median_snr', 'median_snr2'],
    #       dtype='object')
    # All timestamps: 410380
    # Unique timestamps: 4154

    features = ['x','y','z']#,'txpw_actual_dBm','freq_MHz']
    num_node_features = len(features)

    times = data['timestamp']
    unique_times = tuple(set(times))

    graphs = []

    for i in range(len(unique_times)):

        graph = data[data['timestamp']==unique_times[i]]
        transmitters = set(graph['transmitter'])
        receivers = set(graph['receiver'])
        nodes = tuple(transmitters.union(receivers))

        node_index = {}
        for node in nodes:
            node_index[node] = len(node_index)

        num_nodes = len(nodes)

        features = np.zeros((num_nodes, num_node_features))
        labels = np.zeros((num_nodes, num_node_features))
        edge_index_u = []
        edge_index_v = []
        edge_attr = []

        for j in range(len(graph)):
            tx, rx = graph.iloc[j]['transmitter'], graph.iloc[j]['receiver']
            edge_index_u.append(node_index[tx])
            edge_index_v.append(node_index[rx])
            edge_attr.append(float(graph.iloc[j]['measured_path_loss_dB']))

            if 'base' in tx or 'husky' in tx:
                features[node_index[tx]] = graph.iloc[j]['transmitter_x'], graph.iloc[j]['transmitter_y'], graph.iloc[j]['transmitter_z']
            else:
                features[node_index[tx]] = graph.iloc[j]['transmitter_x'], graph.iloc[j]['transmitter_y'], graph.iloc[j]['transmitter_z']

            labels[node_index[tx]] = graph.iloc[j]['transmitter_x'], graph.iloc[j]['transmitter_y'], graph.iloc[j]['transmitter_z']
            # labels[node_index[rx]] = graph.iloc[j]['receiver_x'], graph.iloc[j]['receiver_y'], graph.iloc[j]['receiver_z']

        # print("features:")
        # print(features)
        # print("labels:")
        # print(labels)
        # print("edges:")
        # print(edge_index_u, edge_index_v)
        # print("edge attr:")
        # print(edge_attr)

        features = torch.Tensor(features)
        labels = torch.Tensor(labels)
        edge_index = torch.Tensor([edge_index_u,edge_index_v]).long()
        edge_attr = torch.Tensor(edge_attr)
        graph_data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
        graphs.append(graph_data)

        # if i == 100:
        #     print("stopping early")
        #     break

    data_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    return data_loader

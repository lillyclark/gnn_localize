from process_dataset_new import process_dataset, is_anchor
from models import GCN
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric

data_loader, nodes = process_dataset('datasets/comp1_clean.csv', batch_size=1, threshold=1000, fake_links=True)
num_nodes = len(nodes)
pdist = torch.nn.PairwiseDistance(p=2)

model = GCN(nfeat=num_nodes, nhid=2000, nout=3, dropout=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
loss_fn = torch.nn.MSELoss()

for batch in data_loader:
    print("feature_matrix")
    print(np.round(batch.x.numpy(),2))

    print("actual distances")
    dists = np.zeros((num_nodes,num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dists[i][j] = pdist(batch.y[i].unsqueeze(0), batch.y[j].unsqueeze(0))
            dists[j][i] = pdist(batch.y[i].unsqueeze(0), batch.y[j].unsqueeze(0))
    print(np.round(dists,2))

    print("adjacency")
    adj = torch_geometric.utils.to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr, max_num_nodes=num_nodes)
    print(np.round(adj.numpy(),2))

    for epoch in range(20000):
        model.train()
        optimizer.zero_grad()
        pred = model(batch)
        loss_val = loss_fn(pred[torch.logical_and(batch.present, ~batch.anchors)], batch.y[torch.logical_and(batch.present, ~batch.anchors)])
        loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
        loss_train.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"train:{loss_train.item()}, val:{loss_val.item()}")

print([(node, is_anchor(node)) for node in nodes])
plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), marker='o', label="prediction", c=batch.anchors.numpy())
plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), marker='x', label="actual", c=batch.anchors.numpy())
for u, v in zip(batch.edge_index[0].numpy(), batch.edge_index[1].numpy()):
    plt.plot([batch.y[u][0], batch.y[v][0]], [batch.y[u][1], batch.y[v][1]], color='orange')
plt.legend()
plt.show()

from process_dataset import *
from models import *
from lle_like import solve_like_LLE, neighbors, barycenter_weights, weight_to_mat
from decomposition import normalize_tensor, reduce_rank, denoise_via_SVD
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np

num_nodes = 100
# num_anchors is now what portion to use as training data
num_anchors = 70
threshold = 1.2
n_neighbors = 30

loss_fn = torch.nn.MSELoss()
model = GCN(nfeat=num_nodes, nhid=2000, nout=2, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

plot_num = 5
plot_iter = 10
plot_count = 0
fig, axes = plt.subplots(1,5)

for training_set in range(plot_num*plot_iter):

    np.random.seed(training_set)
    torch.manual_seed(training_set)
    data_loader, num_nodes, noisy_distance_matrix = fake_dataset(num_nodes, num_anchors, threshold=threshold)

    start = time.time()
    for batch in data_loader:

        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            pred = model(batch.x, batch.adj)
            loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
            loss_train.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"epoch:{epoch}, train:{loss_train.item()}")
    print(f"Done in {time.time()-start} seconds.")

    model.eval()
    pred = model(batch.x, batch.adj)
    loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
    print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

    if training_set%plot_iter==0:
        axes[plot_count].scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
        axes[plot_count].scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
        axes[plot_count].set_title(f"err: {np.round(torch.sqrt(loss_test).item(),2)}")
        plot_count += 1
plt.show()

# plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
# plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)
# plt.legend()
# plt.title(f"err: {np.round(torch.sqrt(loss_test).item(),2)}")
# plt.show()
















# import torch_geometric
# import torch
# import numpy as np
# pdist = torch.nn.PairwiseDistance(p=2)
# from torch.nn.functional import normalize
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from models import *
#
# anchor_locs = torch.Tensor([[1,1],[1,3],[3,1],[3,3]])
# num_anchors = 4
# adj = torch.zeros((num_anchors, num_anchors))
# for i in range(num_anchors):
#     for j in range(num_anchors):
#         if i == j:
#             continue
#         adj[i][j] = 1/pdist(anchor_locs[i].unsqueeze(0), anchor_locs[j].unsqueeze(0))
#
# norm_adj = normalize(adj,p=1,dim=1)
# aug_norm_adj = torch.eye(4) + norm_adj
#
# model = MLP(4,4,2,0.0)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
# loss_fn = torch.nn.MSELoss()
#
# for epoch in range(1000):
#     num_train = 50
#     train_data = torch.rand((num_train,2))*5
#     train_features = torch.zeros((num_train,num_anchors))
#     for i in range(num_train):
#         for j in range(num_anchors):
#             train_features[i][j] = pdist(train_data[i].unsqueeze(0), anchor_locs[j].unsqueeze(0))
#
#     model.train()
#     pred = model(train_features,aug_norm_adj)
#
#     # pred_features = torch.zeros((num_train,num_anchors))
#     # for i in range(num_train):
#     #     for j in range(num_anchors):
#     #         pred_features[i][j] = pdist(pred[i].unsqueeze(0), anchor_locs[j].unsqueeze(0))
#
#     loss_train = loss_fn(pred, train_data)
#     # loss_train = loss_fn(pred_features, train_features)
#
#     loss_train.backward()
#     optimizer.step()
#     if epoch%10 == 0:
#         print("loss_train",loss_train.item())
#
# plt.scatter(anchor_locs[:,0], anchor_locs[:,1], label="anchors", marker="X", color="red")
# plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
# plt.scatter(train_data[:,0].detach().numpy(), train_data[:,1].detach().numpy(), label="actual")
# plt.legend()
# plt.title('fingerprint learn to use anchors?')
# plt.show()

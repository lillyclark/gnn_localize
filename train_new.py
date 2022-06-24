from process_dataset_new import process_dataset, fake_dataset
from models import GCN
import torch.optim as optim
import torch
import wandb
import time
import matplotlib.pyplot as plt

num_nodes = 50
num_anchors = 10
threshold = 1

# data_loader, num_nodes = process_dataset('datasets/comp1_clean.csv', batch_size=1, threshold=1000, fake_links=False)
# num_batches = len(data_loader)
# print(num_batches, "batches")
data_loader, num_nodes = fake_dataset(num_nodes, num_anchors, threshold=threshold)

# model = GCN(nfeat=num_nodes, nhid=128, nout=3, dropout=0.01)
model = GCN(nfeat=num_nodes, nhid=2000, nout=2, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
loss_fn = torch.nn.MSELoss()

wandb_log = False
if wandb_log:
    wandb.init(project="GNN-localize", entity="lillyclark", config={})
    wandb.run.name = "new"+"_"+wandb.run.id

start = time.time()
for batch in data_loader:
    for epoch in range(2000):
        # print(batch)
        # print(batch.x)
        # print(batch.edge_index)
        # print(batch.edge_attr)
        # print(batch.y)
        # print(batch.anchors)
        # print(batch.present)

        model.train()
        optimizer.zero_grad()
        pred = model(batch.x, batch.adj)
        loss_val = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
        loss_train = loss_fn(pred[batch.anchors], batch.y[batch.anchors])
        loss_train.backward()
        optimizer.step()
        print(f"epoch:{epoch}, train:{loss_train.item()}, val:{loss_val.item()}")

        if wandb_log:
            wandb.log({"loss_train":loss_train})
            wandb.log({"loss_val":loss_val})

    # print("pred:")
    # print(pred)
    # print("actual:")
    # print(batch.y)

print(f"Done in {time.time()-start} seconds.")
if wandb_log:
    wandb.finish()

model.eval()
pred = model(batch.x, batch.adj)
loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
print(f"test (RMSE):{torch.sqrt(loss_test).item()}")

plt.scatter(pred[:,0].detach().numpy(), pred[:,1].detach().numpy(), label="predicted")
plt.scatter(batch.y[:,0].detach().numpy(), batch.y[:,1].detach().numpy(), label="actual")
plt.legend()
plt.title('My implementation')
plt.show()

from process_dataset import process_dataset
from models import GCN
import torch.optim as optim
import torch
import wandb

data_loader = process_dataset('datasets/comp1_clean.csv', batch_size=32)
num_batches = len(data_loader)
print(num_batches, "batches")

model = GCN(nfeat=3, nhid=2000, nout=3, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
loss_fn = torch.nn.MSELoss()

wandb_log = True
if wandb_log:
    wandb.init(project="GNN-localize", entity="lillyclark", config={})
    wandb.run.name = "comp1_3layers"+"_"+wandb.run.id

for epoch in range(1000):
    for batch in data_loader:
        # TODO hold 30% for testing
        # assert (batch.x == batch.y).all()

        model.train()
        optimizer.zero_grad()

        pred = model(batch)

        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()

        if wandb_log:
            wandb.log({"loss":loss})

    print(epoch, loss.item())

print("")
if wandb_log:
    wandb.finish()

# model_name = 'comp1_100_epochs'
# torch.save(model.state_dict(), "models/"+model_name)

for batch in data_loader:
    model.eval()
    pred = model(batch)
    print(pred)
    print(batch.y)
    break

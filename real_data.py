from process_dataset import load_a_moment
from separate import separate_dataset_find_k1, separate_dataset_multiple_inits, separate_dataset
from lle_like import solve_like_LLE
from decomposition import reduce_rank
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

threshold=50
n_neighbors=8

data_loader, num_nodes, noisy_distance_matrix = load_a_moment(eta=4.11707152,
                                                            Kref=25.110978914824088)

print("MOMENT LOADED")
print(num_nodes,"nodes")
num_batches = len(data_loader)

modelname = "novel"
loss_fn = torch.nn.MSELoss()

if modelname == "novel":
    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        num_anchors = anchor_locs.shape[0]
        start = time.time()
        # X, Y, ff = separate_dataset_find_k1(noisy_distance_matrix,
        #             k0=4,
        #             k1_init=int(135),
        #             step_size=0.33,
        #             n_init=1,
        #             lam=1/(num_nodes**0.5)*2,
        #             # lam=1e-3,
        #             mu=1/(num_nodes**0.5)*2,
        #             # mu=1e-3,
        #             eps=0.001,
        #             plot=False)

        X, Y, ff = separate_dataset_multiple_inits(noisy_distance_matrix,
                    k0=4,
                    k1=136,
                    n_init=10,
                    lam=1/(num_nodes**0.5),
                    mu=1/(num_nodes**0.5),
                    eps=0.001)

        print("ESTIMATED DIST MATRIX")
        print(np.round(X.numpy(),0))

        print("ESTIMATED SPARSITY MATRIX")
        print(np.round(Y.numpy(),0))

        pred = solve_like_LLE(num_nodes,
                    num_anchors,
                    n_neighbors,
                    anchor_locs,
                    X,
                    dont_square=True,
                    anchors_as_neighbors=False)

        # print("PREDICTED LOCATIONS")
        # print(pred)

        loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
        print(f"test (RMSE):{torch.sqrt(loss_test).item()}")
        print(f"{time.time()-start} seconds to solve")

        # k1s = np.linspace(0,300,301)
        # # k1s = np.linspace(0,2,2)
        #
        # ffs = []
        # RMSEs = []
        # for k1 in k1s:
        #     X, Y, ff = separate_dataset_multiple_inits(noisy_distance_matrix, k0=4, k1=int(k1), n_init=1, lam=1/(num_nodes**0.5), mu=1/(num_nodes**0.5), eps=0.001)
        #     print(X)
        #     ffs.append(ff)
        #     pred = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False)
        #     loss_test = loss_fn(pred[batch.nodes], batch.y[batch.nodes])
        #     # if not RMSEs:
        #     #     pass
        #     # elif torch.sqrt(loss_test).item() < min(RMSEs):
        #     print("")
        #     print("k1:",k1)
        #     print(f"test (RMSE):{torch.sqrt(loss_test).item()}")
        #     RMSEs.append(torch.sqrt(loss_test).item())
        #
        # ffs = np.array(ffs)
        # deltas = ffs[:-1]-ffs[1:]
        # relative_deltas = (ffs[:-1]-ffs[1:])/ffs[1:]
        #
        # # c = 24977
        # fig, ax = plt.subplots(1,2)
        # # ax[0].axvline(c)
        # ax[0].plot(k1s,ffs,label="Cost function")
        # ax[0].set_title("Cost function")
        # # ax[1].axvline(c)
        # ax[1].plot(k1s, RMSEs,label="Final actual error")
        # ax[1].set_title("RMSE")
        # plt.suptitle("Performance of alternating min for different guesses of k1 (sparsity)")
        # plt.show()

plt.scatter(pred[:num_anchors,0].detach().numpy(), pred[:num_anchors,1].detach().numpy(), label="predicted a", marker="+",color="blue")#,alpha=0.1)
plt.scatter(batch.y[:num_anchors,0].detach().numpy(), batch.y[:num_anchors,1].detach().numpy(), label="actual a", marker="x",color="orange")#,alpha=0.1)
plt.scatter(pred[num_anchors:,0].detach().numpy(), pred[num_anchors:,1].detach().numpy(), label="predicted",color="blue")#,alpha=0.1)
plt.scatter(batch.y[num_anchors:,0].detach().numpy(), batch.y[num_anchors:,1].detach().numpy(), label="actual",color="orange")#,alpha=0.1)

plt.legend()
plt.title(f"{modelname}: {np.round(torch.sqrt(loss_test).item(),2)}")
plt.show()

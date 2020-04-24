import numpy as np
import matplotlib.pyplot as plt

# Experiment 1
chunk_sizes = [10, 100, 200, 300, 500, 1000]  # number of training instances
steps = [10, 100, 200, 300, 500, 1000]  # prediction time horizon in instances
methods = ["PCALR", "LR"]

data = np.load("results/experiment_1.npy")
fig, ax = plt.subplots(data.shape[2], 1, figsize=(6, 10))

for idx in range(data.shape[2]):
    mdata = data[:,:,idx]

    pl = ax[idx].imshow(mdata, cmap="bwr")
    ax[idx].set_title(methods[idx])
    ax[idx].set_xticklabels([0]+chunk_sizes)
    ax[idx].set_yticklabels([0]+steps)
    ax[idx].set_xlabel("Chunk size")
    ax[idx].set_ylabel("Prediction horizon")
    plt.colorbar(pl,ax=ax[idx])

    for i, row in enumerate(mdata):
        for j, value in enumerate(row):
            print(i,j,value)
            ax[idx].text(j,i,"%.2f" % value, horizontalalignment='center',verticalalignment='center', fontsize=8)



plt.tight_layout()
plt.savefig("figures/experiment_1.png")

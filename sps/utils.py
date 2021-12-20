import numpy as np, matplotlib, matplotlib.pyplot as plt 

def kill_ticks(ax: matplotlib.axes.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

def imshow(*ims: np.ndarray):
    n = len(ims)
    if n == 1:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(ims[0], cmap='gray')
        kill_ticks(ax)
    else:
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        for im, ax in zip(ims, axes):
            ax.imshow(im, cmap='gray')
            kill_ticks(ax)

    plt.tight_layout(); plt.show(); plt.close()
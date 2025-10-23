import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import matplotlib
import os

from cfg import parse_args
args = parse_args()

def shape_image(data, ax, ptype="source"):
    xlim = ylim = zlim = [-.14,.14]
    view_init = (90,-90,'z')
    dx, dy, dz = (0.02,-0.1,0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(view_init[0],view_init[1],vertical_axis=view_init[2])
    if ptype == "target":    
        ax.scatter(data[::5,0]+dx,data[::5,1]+dy,data[::5,2]+dz,alpha=1.,s=2,marker='.')#, c='C2')
    elif ptype == "source":
        ax.scatter(data[::5,0]+dx,data[::5,1]+dy,data[::5,2]+dz,alpha=1.,s=2,marker='.', c='C1')
    
    # ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(True)
    # ax.axis('off')

matplotlib.rcParams.update({'font.size': 22})
target = torch.load("shape_data_bai/bunny_noise.pt")

# ckpts = [i.split('_')[2].split('-')[1] for i in os.listdir('results_icp')]
method = args.method
style = args.plot_style
plot_iters = [0, 100, 200, 300, -1]
os.makedirs(f"shape_figs/", exist_ok=True)

if not os.path.exists(f"results/{method}"):
    print("No results found, exiting.")
    exit()

base_dir = f'results/{method}'
if style == 0: # show progress
    ckpts = os.listdir(base_dir)
    assert len(ckpts) == 1
    rows = len(ckpts); cols = len(plot_iters)
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    for row_id, ckpt in enumerate(ckpts):
        data_results = np.load(f"{base_dir}/{ckpt}")
        for col_id, n_iter in enumerate(plot_iters):
            ax_id = row_id * cols + col_id + 1
            ax = fig.add_subplot(rows, cols, ax_id, projection='3d')
            if col_id == len(plot_iters) - 1:
                shape_image(target, ax=ax, ptype='target')
                continue
            transformed_source_data = data_results["X"][n_iter]  
            shape_image(transformed_source_data, ax=ax, ptype='source')

elif style == 1: # show different config
    ckpts = os.listdir(base_dir)
    cols = len(ckpts); rows = 1
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    for col_id, ckpt in enumerate(ckpts):
        data_results = np.load(f"{base_dir}/{ckpt}")
        ax_id = col_id + 1
        ax = fig.add_subplot(rows, cols, ax_id, projection='3d')
        transformed_source_data = data_results["X"][-1]  
        hypers = ckpt.split("_")[0]
        reg, k = hypers.split("-")
        shape_image(transformed_source_data, ax=ax, ptype='source')
        ax.set_title(f"{reg}-{k}")

elif style == 2: # for tuning
    ckpts = os.listdir(base_dir)
    rows = len(ckpts); cols = len(plot_iters)
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    for row_id, ckpt in enumerate(ckpts):
        data_results = np.load(f"{base_dir}/{ckpt}")
        for col_id, n_iter in enumerate(plot_iters):
            ax_id = row_id * cols + col_id + 1
            ax = fig.add_subplot(rows, cols, ax_id, projection='3d')
            if col_id == len(plot_iters) - 1:
                shape_image(target, ax=ax, ptype='target')
                ax.set_title("Target")
                continue

            transformed_source_data = data_results["X"][n_iter]  
            hypers = ckpt.split("_")[0]
            reg, k = hypers.split("-")
            shape_image(transformed_source_data, ax=ax, ptype='source')
            if col_id == 0:
                ax.set_title(f"{reg}-{k}")
            if row_id == 0 and col_id > 0:
                ax.set_title(f"Iter {n_iter}")

plt.tight_layout()
plt.subplots_adjust(wspace=-0.6)#, hspace=-0.2)
plt.savefig(f"shape_figs/{method}_style{style}.png")

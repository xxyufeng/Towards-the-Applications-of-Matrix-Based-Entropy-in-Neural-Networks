import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from matplotlib import cm as colormaps

def plot_PI(result_path, save_path):
    cmap_list = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Blues', 'Greens', 'Oranges', 'Reds', 'Purples',
        'cool', 'winter', 'autumn', 'spring', 'summer',
        'turbo', 'nipy_spectral', 'gist_rainbow', 'rainbow', 'gist_stern',
        'YlOrBr', 'YlOrRd', 'OrRd', 'Oranges', 'Reds'
    ]
    np.random.seed(0)
    cid = np.random.permutation(len(cmap_list))

    # Load MI results from CSV
    df = pd.read_csv(result_path)
    cols = ['layer_FMSA_mid: I(X;T)', 'layer_FMSA_mid: I(T;Y)',
        'layer_FMSA: I(X;T)', 'layer_FMSA: I(T;Y)',
        'layer_CLS: I(X;T)',  'layer_CLS: I(T;Y)',
        'layer_LAST: I(X;T)', 'layer_LAST: I(T;Y)']

    n_iter = int(df['Iteration'].max()) + 1

    mi = np.zeros((1, n_iter, 4, 2), dtype=float)
    for _, row in df.iterrows():
        it = int(row['Iteration'])
        vals = row[cols].to_numpy(dtype=float)
        mi[0, it] = vals.reshape(4, 2)
    N, n_iters, n_layers = len(mi), len(mi[0]), len(mi[0][0])
    print(N, n_iters, n_layers, len(mi[0][0][0]))
    c_lab = [cmap_list[cid[i]] for i in range(n_layers)]
    xy1 = np.zeros((n_iters, n_layers, 2))
    for i in range(n_iters):
        for l in range(n_layers):
            xy1[i, l, :] = np.array(mi[0][i][l])

    mask = ~np.isnan(xy1).any(axis=(1, 2))
    xy1 = xy1[mask].transpose(1, 0, 2)

    c_lab = ['Greens', 'Oranges', 'Purples', 'Reds']
    layer = ['FMSA_MID', 'FMSA', 'CLS', 'LAST']

    plt.figure(1, figsize=(10, 6))
    for m, hei in enumerate(layer):
        plt.scatter(xy1[m, :, 0], xy1[m, :, 1], cmap=c_lab[m], c=np.arange(0,xy1.shape[1], 1), edgecolor=c_lab[m][:-1], s=30, alpha=0.8)
        if hei == layer[0]:
            plt.colorbar(pad=-0.12)
        elif hei == layer[-1]:
            plt.colorbar(ticks=[], pad=0.001)
        else:
            plt.colorbar(ticks=[], pad=-0.12)
    for m, hei in enumerate(layer):
        plt.scatter(xy1[m, -1, 0], xy1[m, -1, 1], c=c_lab[m][:-1], label='Layer {}'.format(hei), s=30)
        plt.scatter(xy1[m, 2500, 0], xy1[m, 2500, 1],c='k', s=300, marker='^')
        # plt.scatter(xy1[m, 7500, 0], xy1[m, 7500, 1], c=c_lab[m][:-1], label='Layer {} (Iter 7500)'.format(hei), s=500, marker='^')
    plt.legend(facecolor='white')
    plt.xlabel('MI(X,T)')
    plt.ylabel('MI(T,Y)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


    # try:
    #     from matplotlib.animation import FuncAnimation, PillowWriter
    #     from matplotlib.lines import Line2D

    #     n_layers_plot = xy1.shape[0]
    #     n_points = xy1.shape[1]
    #     base_colors = plt.get_cmap('tab10').colors
    #     layer_names = ['FMSA_MID', 'FMSA', 'CLS', 'LAST']

    #     fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
    #     ax_anim.set_xlabel('MI(X,T)')
    #     ax_anim.set_ylabel('MI(T,Y)')
    #     ax_anim.set_title('MI trajectories over iterations')

    #     # Initialize empty scatter objects for each layer
    #     scatters = []
    #     for m in range(n_layers_plot):
    #         sc = ax_anim.scatter([], [], color=base_colors[m % len(base_colors)], s=40, alpha=0.9)
    #         scatters.append(sc)

    #     # Legend using proxy artists
    #     proxies = [Line2D([0], [0], marker='o', color='w', markerfacecolor=base_colors[i % len(base_colors)], markersize=8, label=layer_names[i]) for i in range(n_layers_plot)]
    #     ax_anim.legend(handles=proxies, loc='upper left', facecolor='white')

    #     # Axis limits set from data for stability
    #     all_x = xy1[:, :, 0].reshape(-1)
    #     all_y = xy1[:, :, 1].reshape(-1)
    #     ax_anim.set_xlim(np.nanmin(all_x) - 0.05 * abs(np.nanmin(all_x)), np.nanmax(all_x) + 0.05 * abs(np.nanmax(all_x)))
    #     ax_anim.set_ylim(np.nanmin(all_y) - 0.05 * abs(np.nanmin(all_y)), np.nanmax(all_y) + 0.05 * abs(np.nanmax(all_y)))

    #     FRAME_COUNT = 200           
    #     INTERVAL_MS = 100             
    #     indices = np.linspace(0, n_points-1, FRAME_COUNT).astype(int)  

    #     def update_by_index(i):
    #         data_idx = indices[i]
    #         for m in range(n_layers_plot):
    #             xs = xy1[m, :data_idx+1, 0]    
    #             ys = xy1[m, :data_idx+1, 1]
    #             scatters[m].set_offsets(np.column_stack((xs, ys)))
    #         ax_anim.set_title(f'iter={data_idx}')
    #         return scatters

    #     anim = FuncAnimation(fig_anim, update_by_index, frames=len(indices), interval=10, blit=False)

    #     # Save animation as GIF next to static image
    #     save_path_anim = save_path.replace('.png', '_anim.gif') if save_path.endswith('.png') else save_path + '_anim.gif'
    #     try:
    #         fps = max(1, int(1000 / INTERVAL_MS))
    #         writer = PillowWriter(fps=fps)
    #         # When frames > data points, update() maps frame -> data index, so animation still valid
    #         anim.save(save_path_anim, writer=writer)
    #     except Exception:
    #         # If saving GIF fails, try MP4 (requires ffmpeg)
    #         try:
    #             save_path_mp4 = save_path_anim.replace('.gif', '.mp4')
    #             anim.save(save_path_mp4, fps=fps)
    #         except Exception:
    #             pass

    #     plt.close(fig_anim)
    # except Exception:
    #     pass

def plot_training_history(history_path='training_history.csv',save_path='training_history.png'):
    """Plot training/validation loss and accuracy curves"""
    plt.figure(figsize=(10, 6))

    try:
        history = pd.read_csv(history_path, header=0)
    except Exception as e:
        raise RuntimeError(f"Failed to read history CSV '{history_path}': {e}")

    def find_col(candidates):
        cols = {c.lower().replace(' ', '').replace('_', ''): c for c in history.columns}
        for cand in candidates:
            key = cand.lower().replace(' ', '').replace('_', '')
            if key in cols:
                return cols[key]
        return None

    train_loss_col = find_col(['train_loss', 'trainloss', 'loss_train', 'losstrain', 'train loss'])
    val_loss_col = find_col(['val_loss', 'valid_loss', 'valueloss', 'val loss', 'validation_loss', 'validationloss'])
    train_acc_col  = find_col(['train_acc', 'trainacc', 'accuracy_train', 'acc_train', 'train acc'])
    val_acc_col    = find_col(['val_acc', 'valacc', 'accuracy_val', 'acc_val', 'val acc', 'validation_acc'])

    if train_loss_col is None or val_loss_col is None:
        raise ValueError(f"Could not find loss columns in '{history_path}'. Available columns: {list(history.columns)}")
    if train_acc_col is None or val_acc_col is None:
        has_acc = False
    else:
        has_acc = True

    epoch_col = find_col(['epoch', 'ep', 'step'])
    if epoch_col is not None:
        x = history[epoch_col].to_numpy()
        x = x * 195
        xlabel = epoch_col
    else:
        x = np.arange(len(history))
        xlabel = 'Epoch'

    # Loss curve
    plt.subplot(1, 1, 1)
    plt.plot(x, history[train_loss_col], label='Train Loss', linewidth=2, color='blue')
    plt.plot(x, history[val_loss_col],   label='Val Loss',   linewidth=2, color='red', linestyle='--')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Training & Validation Loss', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # # Accuracy curve（若有）
    # plt.subplot(1, 2, 2)
    # if has_acc:
    #     plt.plot(x, history[train_acc_col], label='Train Accuracy', linewidth=2, color='blue')
    #     plt.plot(x, history[val_acc_col],   label='Val Accuracy',   linewidth=2, color='red', linestyle='--')
    #     plt.xlabel(xlabel, fontsize=10)
    #     plt.ylabel('Accuracy', fontsize=10)
    #     plt.title('Training & Validation Accuracy', fontsize=12)
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    # else:
    #     plt.text(0.5, 0.5, 'Accuracy columns not found in history file', horizontalalignment='center', verticalalignment='center')
    #     plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    

plot_PI('vit_cifar10_mi_results_1116.csv', 'MI_ViT_CIFAR10_epochs100_batch256_lr1e-4_wd1e-5_1116.png')
# plot_training_history('vit_cifar10_training_history_1116.csv', 'training_history.png')
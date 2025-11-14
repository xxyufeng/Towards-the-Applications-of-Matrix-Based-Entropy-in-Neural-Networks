import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from matplotlib import cm as colormaps

def plot_PI(result_path, save_path):
    cmap_list = [
        # 5 套现代 perceptually-uniform
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        # 单色调彩色
        'Blues', 'Greens', 'Oranges', 'Reds', 'Purples',
        # 暖 / 冷 渐变
        'cool', 'winter', 'autumn', 'spring', 'summer',
        # 多色线性
        'turbo', 'nipy_spectral', 'gist_rainbow', 'rainbow', 'gist_stern',
        # 黄-红-褐 系列
        'YlOrBr', 'YlOrRd', 'OrRd', 'Oranges', 'Reds'
    ]
    np.random.seed(0)
    cid = np.random.permutation(len(cmap_list))

    # Load MI results from CSV
    df = pd.read_csv(result_path)
    cols = ['layer_FMSA: I(X;T)', 'layer_FMSA: I(T;Y)',
        'layer_CLS: I(X;T)',  'layer_CLS: I(T;Y)',
        'layer_LAST: I(X;T)', 'layer_LAST: I(T;Y)']

    n_iter = int(df['Iteration'].max()) + 1

    mi = np.zeros((1, n_iter, 3, 2), dtype=float)
    for _, row in df.iterrows():
        it = int(row['Iteration'])
        vals = row[cols].to_numpy(dtype=float)
        mi[0, it] = vals.reshape(3, 2)

    N, n_iters, n_layers = len(mi), len(mi[0]), len(mi[0][0])
    print(N, n_iters, n_layers, len(mi[0][0][0]))
    c_lab = [cmap_list[cid[i]] for i in range(n_layers)]
    xy1 = np.zeros((n_iters, n_layers, 2))
    for i in range(n_iters):
        for l in range(n_layers):
            xy1[i, l, :] = np.array(mi[0][i][l])

    mask = ~np.isnan(xy1).any(axis=(1, 2))
    xy1 = xy1[mask].transpose(1, 0, 2)

    c_lab = ['Greens', 'Oranges', 'Purples']
    layer = ['FMSA', 'CLS', 'LAST']

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
    plt.legend(facecolor='white')
    plt.xlabel('MI(X,T)')
    plt.ylabel('MI(T,Y)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    # --- 动图：随 iterations 从小到大逐步显示散点（不使用渐变色，也不需要 colorbar） ---
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.lines import Line2D

        n_layers_plot = xy1.shape[0]
        n_points = xy1.shape[1]
        base_colors = plt.get_cmap('tab10').colors
        layer_names = ['FMSA', 'CLS', 'LAST']

        fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
        ax_anim.set_xlabel('MI(X,T)')
        ax_anim.set_ylabel('MI(T,Y)')
        ax_anim.set_title('MI trajectories over iterations')

        # Initialize empty scatter objects for each layer
        scatters = []
        for m in range(n_layers_plot):
            sc = ax_anim.scatter([], [], color=base_colors[m % len(base_colors)], s=40, alpha=0.9)
            scatters.append(sc)

        # Legend using proxy artists
        proxies = [Line2D([0], [0], marker='o', color='w', markerfacecolor=base_colors[i % len(base_colors)], markersize=8, label=layer_names[i]) for i in range(n_layers_plot)]
        ax_anim.legend(handles=proxies, loc='upper left', facecolor='white')

        # Axis limits set from data for stability
        all_x = xy1[:, :, 0].reshape(-1)
        all_y = xy1[:, :, 1].reshape(-1)
        ax_anim.set_xlim(np.nanmin(all_x) - 0.05 * abs(np.nanmin(all_x)), np.nanmax(all_x) + 0.05 * abs(np.nanmax(all_x)))
        ax_anim.set_ylim(np.nanmin(all_y) - 0.05 * abs(np.nanmin(all_y)), np.nanmax(all_y) + 0.05 * abs(np.nanmax(all_y)))

        FRAME_COUNT = 200            # 减少到 200 帧（按需调小/调大）
        INTERVAL_MS = 100             # 每帧间隔 50 毫秒
        indices = np.linspace(0, n_points-1, FRAME_COUNT).astype(int)  # 采样索引

        def update_by_index(i):
            data_idx = indices[i]
            for m in range(n_layers_plot):
                xs = xy1[m, :data_idx+1, 0]    # 使用累计显示（或只显示到 data_idx）
                ys = xy1[m, :data_idx+1, 1]
                scatters[m].set_offsets(np.column_stack((xs, ys)))
            ax_anim.set_title(f'iter={data_idx}')
            return scatters

        anim = FuncAnimation(fig_anim, update_by_index, frames=len(indices), interval=10, blit=False)

        # Save animation as GIF next to static image
        save_path_anim = save_path.replace('.png', '_anim.gif') if save_path.endswith('.png') else save_path + '_anim.gif'
        try:
            fps = max(1, int(1000 / INTERVAL_MS))
            writer = PillowWriter(fps=fps)
            # When frames > data points, update() maps frame -> data index, so animation still valid
            anim.save(save_path_anim, writer=writer)
        except Exception:
            # If saving GIF fails, try MP4 (requires ffmpeg)
            try:
                save_path_mp4 = save_path_anim.replace('.gif', '.mp4')
                anim.save(save_path_mp4, fps=fps)
            except Exception:
                pass

        plt.close(fig_anim)
    except Exception:
        # 动画为可选功能，失败不影响静态图
        pass
    

plot_PI('vit_cifar10_mi_results.csv', 'MI_ViT_CIFAR10_epochs100_batch256_lr1e-4_wd1e-5.png')
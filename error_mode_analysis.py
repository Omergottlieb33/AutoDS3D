"""Break down localization failures vs aberration RMS.

For every ground-truth emitter, the Hungarian match (3D) to the predictions puts it in one of:
matched within 100 nm, 100-250 nm, 250-500 nm, > 500 nm, or no prediction (more emitters than
predictions). Also reports recall by emitter density, and lateral-only vs 3D recall.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
from DS3Dplus.ds3d_utils import Volume2XYZ
from evaluation import get_image_tensor
from aberration_sweep_evaluation import (ABR_CLEAN, ABR_NOISY, ORIG_CLEAN, ORIG_NOISY, SERIES_COLORS, SEQUENTIAL_BLUES,
                                         UM_TO_NM, default_run_labels, find_levels, load_net)

DISTANCE_BINS_UM = [0.1, 0.25, 0.5]
CATEGORIES = ['< 100 nm', '100–250 nm', '250–500 nm', '> 500 nm', 'no prediction']
DENSITY_BINS = [0, 10, 20, 35]
DENSITY_LABELS = ['1–10', '11–20', '21–35']


def get_args():
    parser = argparse.ArgumentParser(description="Localization error modes vs aberration RMS")
    parser.add_argument('--weights_paths', type=str, nargs='+', required=True, help='.pt checkpoints to analyse')
    parser.add_argument('--labels', type=str, nargs='+', default=None, help='Name per checkpoint (file names, titles)')
    parser.add_argument('--data_root', type=str, required=True, help='Aberration sweep folder')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--noise', action='store_true', help=f"Use '{ABR_NOISY}' images instead of '{ABR_CLEAN}'")
    parser.add_argument('--threshold', type=float, default=40, help='Decode threshold')
    parser.add_argument('--blob_r', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def match_emitters(xyz_gt, xyz_pred):
    """Per GT emitter: 3D and lateral distance (um) to its Hungarian match; NaN when unmatched."""
    dist_3d = np.full(len(xyz_gt), np.nan)
    dist_xy = np.full(len(xyz_gt), np.nan)
    if xyz_pred is not None:
        cost = pairwise_distances(xyz_pred, xyz_gt)
        rec, gt = linear_sum_assignment(cost)
        dist_3d[gt] = cost[rec, gt]
        cost_xy = pairwise_distances(xyz_pred[:, :2], xyz_gt[:, :2])
        rec, gt = linear_sum_assignment(cost_xy)
        dist_xy[gt] = cost_xy[rec, gt]
    return dist_3d, dist_xy


def analyse_model(net, levels, data_root, noise, threshold, blob_r, device):
    abr_variant, baseline_variant = (ABR_NOISY, ORIG_NOISY) if noise else (ABR_CLEAN, ORIG_CLEAN)
    _, first_sub, first_loc, first_params = levels[0]
    jobs = [(0.0, first_sub, baseline_variant, first_loc, first_params)] + \
        [(rms, sub, abr_variant, loc, params) for rms, sub, loc, params in levels]
    rows = []
    for rms, sub, variant, localizations, param_dict in jobs:
        decode = Volume2XYZ(dict(param_dict, device=device, blob_r=blob_r, threshold=threshold))
        imgs_path = os.path.join(data_root, sub, 'x', variant)
        for img_name in tqdm(sorted(f for f in os.listdir(imgs_path) if f.endswith('.tif')), desc=f'{sub}/{variant}'):
            xyz_gt = localizations[img_name]['xyzps'][:, :3]
            with torch.no_grad():
                volume = net(get_image_tensor(imgs_path, img_name, dict(param_dict, device=device)))
            xyz_pred, _ = decode(volume)
            dist_3d, dist_xy = match_emitters(xyz_gt, xyz_pred)
            for d3, dxy in zip(dist_3d, dist_xy):
                rows.append({'Aberration RMS': rms, 'Image': img_name, 'Emitters in image': len(xyz_gt),
                             'Distance 3D (nm)': d3 * UM_TO_NM, 'Distance xy (nm)': dxy * UM_TO_NM})
    return pd.DataFrame(rows)


def summarize_error_modes(per_emitter: pd.DataFrame) -> pd.DataFrame:
    d3 = per_emitter['Distance 3D (nm)'] / UM_TO_NM
    edges = [0] + DISTANCE_BINS_UM + [np.inf]
    per_emitter = per_emitter.assign(
        Category=pd.cut(d3, edges, labels=CATEGORIES[:-1], right=False).cat.add_categories(CATEGORIES[-1])
        .fillna(CATEGORIES[-1]),
        Density=pd.cut(per_emitter['Emitters in image'], DENSITY_BINS, labels=DENSITY_LABELS))
    grouped = per_emitter.groupby('Aberration RMS')
    summary = pd.crosstab(per_emitter['Aberration RMS'], per_emitter['Category'], normalize='index')
    summary = summary.reindex(columns=CATEGORIES, fill_value=0.0)
    summary.columns = [f'Fraction {c}' for c in CATEGORIES]
    summary['Recall 3D < 100 nm'] = grouped['Distance 3D (nm)'].apply(lambda d: (d < 100).mean())
    summary['Recall xy < 100 nm'] = grouped['Distance xy (nm)'].apply(lambda d: (d < 100).mean())
    density = per_emitter.groupby(['Aberration RMS', 'Density'], observed=True)['Distance 3D (nm)'] \
        .apply(lambda d: (d < 100).mean()).unstack()
    density.columns = [f'Recall 3D < 100 nm, {c} emitters' for c in density.columns]
    return summary.join(density).reset_index()


def style_axis(ax, title, ink, muted):
    ax.set_xlabel('Aberration RMS (rad)', color=ink)
    ax.set_title(title, color=ink, loc='left')
    ax.set_ylim(0, 1.02)
    ax.grid(axis='y', color='#e5e4e0', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=muted)


def plot_error_modes(summary: pd.DataFrame, save_path: str, title: str):
    ink, muted = '#0b0b0b', '#52514e'
    x = summary['Aberration RMS'].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # 1. where every ground-truth emitter ends up (part-to-whole, ordered good -> bad)
    ax = axes[0]
    fills = ['#104281', '#2a78d6', '#6da7ec', '#b7d3f6', '#c3c2b7']
    ax.stackplot(x, [summary[f'Fraction {c}'] for c in CATEGORIES], colors=fills, labels=CATEGORIES,
                 edgecolor='white', linewidth=1)
    style_axis(ax, 'Ground-truth emitters by match distance', ink, muted)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of emitters', color=ink)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='none', fontsize=8)

    # 2. recall by emitter density (ordered bins -> sequential ramp)
    ax = axes[1]
    for label, color in zip(DENSITY_LABELS, [SEQUENTIAL_BLUES[1], SEQUENTIAL_BLUES[5], SEQUENTIAL_BLUES[9]]):
        col = f'Recall 3D < 100 nm, {label} emitters'
        ax.plot(x, summary[col], color=color, linewidth=2, marker='o', markersize=5, markeredgecolor='white',
                markeredgewidth=1, label=f'{label} emitters / image')
    style_axis(ax, 'Recall (3D < 100 nm) by emitter density', ink, muted)
    ax.legend(loc='lower left', frameon=False, fontsize=8)

    # 3. lateral-only vs 3D recall: the gap is emitters right in xy but wrong in z
    ax = axes[2]
    for col, label, color in [('Recall xy < 100 nm', 'within 100 nm laterally (any z)', SERIES_COLORS[1]),
                              ('Recall 3D < 100 nm', 'within 100 nm in 3D', SERIES_COLORS[0])]:
        ax.plot(x, summary[col], color=color, linewidth=2, marker='o', markersize=5, markeredgecolor='white',
                markeredgewidth=1, label=label)
    style_axis(ax, 'Lateral-only vs 3D recall', ink, muted)
    ax.legend(loc='lower left', frameon=False, fontsize=8)

    fig.suptitle(title, color=ink)
    fig.text(0.5, -0.03, 'Each ground-truth emitter is matched to a prediction by the Hungarian algorithm (3D, or xy '
             'for the lateral curve); "no prediction" = more emitters than predictions. RMS 0 = unaberrated baseline.',
             ha='center', color=muted, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    args = get_args()
    labels = args.labels or default_run_labels(args.weights_paths)
    if len(labels) != len(args.weights_paths):
        raise ValueError(f"Got {len(labels)} labels for {len(args.weights_paths)} checkpoints")
    os.makedirs(args.save_dir, exist_ok=True)
    levels = find_levels(args.data_root)
    tag = 'noisy' if args.noise else 'clean'
    for label, weights_path in zip(labels, args.weights_paths):
        net, checkpoint_path = load_net(weights_path, args.device)
        per_emitter = analyse_model(net, levels, args.data_root, args.noise, args.threshold, args.blob_r, args.device)
        summary = summarize_error_modes(per_emitter)
        name = label.replace('/', '_')
        summary.to_csv(os.path.join(args.save_dir, f'error_modes_{name}_{tag}.csv'), index=False)
        plot_error_modes(summary, os.path.join(args.save_dir, f'error_modes_{name}_{tag}.png'),
                         f'{label} — error modes on {ABR_NOISY if args.noise else ABR_CLEAN}, '
                         f'threshold {args.threshold:g}')
        with pd.option_context('display.width', 220, 'display.max_columns', None, 'display.precision', 3):
            print(f'\n{label}\n{summary.to_string(index=False)}')
    print(f"Saved error-mode results to {args.save_dir}")


if __name__ == "__main__":
    main()

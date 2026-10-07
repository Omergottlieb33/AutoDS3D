import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from DS3Dplus.ds3d_utils import Volume2XYZ, calc_jaccard_rmse
from DS3Dplus.ds3d_utils import LON as Net
from evaluation import get_image_tensor, get_subfolders, find_latest_pt_file

# image sub-folders inside <level>/x/
ABR_NOISY, ABR_CLEAN = 'abr', 'abr_clean'
ORIG_NOISY, ORIG_CLEAN = 'orig_noisy', 'orig_clean'
UM_TO_NM = 1e3  # labels and predictions are in microns
# categorical colors, assigned to runs in this fixed order
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
# sequential blue ramp (light -> dark) for ordered values such as decode thresholds
SEQUENTIAL_BLUES = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95',
                    '#104281', '#0d366b']
METRICS = ['Jaccard Index', 'Precision', 'Recall', 'RMSE_xy (nm)', 'RMSE_z (nm)']


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a DS3D+ model on datasets with increasing aberration, "
                    "or overlay saved sweeps with --compare")
    parser.add_argument('--weights_path', type=str,
                        help='Path to a .pt checkpoint, or a folder (latest .pt is used)')
    parser.add_argument('--data_root', type=str,
                        help='Folder with one sub-folder per aberration level (each with x/, y.pickle, param.pickle)')
    parser.add_argument('--noise', action='store_true',
                        help=f"Infer on aberrated + noisy images ('{ABR_NOISY}') instead of aberration only ('{ABR_CLEAN}')")
    parser.add_argument('--baseline_variant', type=str, default=None, choices=[ORIG_CLEAN, ORIG_NOISY],
                        help=f"Unaberrated baseline images from the first level. Default: '{ORIG_NOISY}' with --noise, else '{ORIG_CLEAN}'")
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output folder. Default: the folder containing the model weights')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--blob_r', type=float, default=2.0,
                        help='Blob radius for evaluation')
    parser.add_argument('--threshold', type=float, default=40,
                        help='Decode threshold for the main plot (and the --compare overlays)')
    parser.add_argument('--thresholds', type=float, nargs='+', default=None,
                        help='Extra decode thresholds evaluated from the same network output; adds a threshold-sweep plot')
    parser.add_argument('--jaccard_threshold', type=float, default=0.1,
                        help='Matching radius for Jaccard (microns)')
    # overlay mode
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                        help='results_summary_*.csv files to overlay on one plot (skips evaluation)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Legend label per --compare file. Default: weights folder names')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Comparison plot path. Default: <common folder>/aberration_sweep_comparison_<clean|noisy|mixed>.png')
    parser.add_argument('--no_std', action='store_true',
                        help='Hide the ±1 std bands in the comparison plot')
    args = parser.parse_args()
    if args.compare is None and (args.weights_path is None or args.data_root is None):
        parser.error('--weights_path and --data_root are required unless --compare is given')
    return args


def load_net(weights_path: str, device: str):
    checkpoint_path = find_latest_pt_file(weights_path) if os.path.isdir(weights_path) else weights_path
    # Allowlist the model class for secure unpickling on PyTorch >= 2.6
    try:
        torch.serialization.add_safe_globals([Net])
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()
    print(f"Loaded model from {checkpoint_path}")
    return net, checkpoint_path


def load_level(level_dir: str):
    with open(os.path.join(level_dir, 'y.pickle'), 'rb') as handle:
        localizations = pickle.load(handle)
    with open(os.path.join(level_dir, 'param.pickle'), 'rb') as handle:
        param_dict = pickle.load(handle)
    return localizations, param_dict


def find_levels(data_root: str) -> list:
    """[(total_rms, sub_folder, localizations, param_dict)] for every aberration level, ordered by RMS."""
    levels = []
    for sub in get_subfolders(data_root):
        level_dir = os.path.join(data_root, sub)
        if not os.path.isfile(os.path.join(level_dir, 'param.pickle')):
            continue
        localizations, param_dict = load_level(level_dir)
        levels.append((param_dict['total_rms'], sub, localizations, param_dict))
    if not levels:
        raise FileNotFoundError(f"No aberration levels with param.pickle found in {data_root}")
    return sorted(levels, key=lambda level: level[0])


def evaluate_images(net, imgs_path, localizations, param_dict, device, blob_r, thresholds, jaccard_threshold,
                    desc=None):
    """Per-image detection and localization metrics (RMSE in nm), decoded at each threshold."""
    decoders = {t: Volume2XYZ(dict(param_dict, device=device, blob_r=blob_r, threshold=t)) for t in thresholds}
    img_names = sorted([f for f in os.listdir(imgs_path) if f.endswith('.tif')],
                       key=lambda x: int(os.path.splitext(x)[0]))
    rows = []
    for img_name in tqdm(img_names, desc=desc):
        xyz_gt = localizations[img_name]['xyzps'][:, :-1]
        im_tensor = get_image_tensor(imgs_path, img_name, dict(param_dict, device=device))
        with torch.no_grad():
            volume_pred = net(im_tensor)
        for threshold, volume2xyz in decoders.items():
            xyz_pred, _ = volume2xyz(volume_pred)
            jaccard, rmse_xy, rmse_z, _ = calc_jaccard_rmse(xyz_gt, xyz_pred, jaccard_threshold)
            n_gt = len(xyz_gt)
            n_pred = 0 if xyz_pred is None else len(xyz_pred)
            # invert jaccard = TP / (n_pred + n_gt - TP)
            tp = int(round(jaccard * (n_pred + n_gt) / (1 + jaccard)))
            rows.append({
                'Image': img_name,
                'Threshold': threshold,
                'Jaccard Index': jaccard,
                'Precision': tp / n_pred if n_pred else np.nan,
                'Recall': tp / n_gt,
                'RMSE_xy (nm)': np.nan if rmse_xy is None else rmse_xy * UM_TO_NM,
                'RMSE_z (nm)': np.nan if rmse_z is None else rmse_z * UM_TO_NM,
                'Num GT': n_gt,
                'Num Pred': n_pred,
                'TP': tp,
                'Empty Prediction': xyz_pred is None,
            })
    return pd.DataFrame(rows)


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    keys = ['Condition', 'Aberration RMS', 'Threshold']
    grouped = results_df.groupby(keys, sort=False)
    summary = grouped[METRICS].agg(['mean', 'std'])
    summary.columns = [f'{m} {stat}' for m, stat in summary.columns]
    counts = grouped[['Num GT', 'Num Pred', 'TP', 'Empty Prediction']].sum()
    # pooled over all emitters of the level, not averaged per image
    summary['Pooled Precision'] = counts['TP'] / counts['Num Pred'].where(counts['Num Pred'] > 0)
    summary['Pooled Recall'] = counts['TP'] / counts['Num GT']
    summary['Num Images'] = grouped.size()
    summary['Num Empty Predictions'] = counts['Empty Prediction'].astype(int)
    return summary.reset_index()


def at_threshold(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rows decoded at one threshold. Summaries saved before threshold sweeps have no Threshold column."""
    if 'Threshold' not in summary.columns:
        return summary
    if not np.isclose(summary['Threshold'], threshold).any():
        raise ValueError(f"Threshold {threshold} not in summary (available: {sorted(summary['Threshold'].unique())})")
    return summary[np.isclose(summary['Threshold'], threshold)].reset_index(drop=True)


def plot_metrics_vs_aberration(summaries: dict, save_path: str, title: str, show_std: bool = True):
    """summaries: {run label: summary DataFrame}. One run also draws its baseline as a dashed line."""
    if len(summaries) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} runs can be overlaid, got {len(summaries)}")
    ink, muted = '#0b0b0b', '#52514e'
    metrics = [('Jaccard Index', 'Jaccard index'),
               ('RMSE_xy (nm)', 'Lateral RMSE (nm)'),
               ('RMSE_z (nm)', 'Axial RMSE (nm)')]
    single_run = len(summaries) == 1
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (col, label) in zip(axes, metrics):
        for color, (run_label, summary) in zip(SERIES_COLORS, summaries.items()):
            x = summary['Aberration RMS'].to_numpy()
            mean = summary[f'{col} mean'].to_numpy()
            if show_std:
                std = summary[f'{col} std'].to_numpy()
                lower, upper = mean - std, mean + std
                if col == 'Jaccard Index':
                    lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
                ax.fill_between(x, lower, upper, color=color, alpha=0.15 if single_run else 0.1, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, marker='o', markersize=6,
                    markeredgecolor='white', markeredgewidth=1.5, label=run_label)
            baseline = summary[summary['Condition'] == 'baseline']
            if single_run and not baseline.empty:
                ax.axhline(baseline[f'{col} mean'].iloc[0], color=muted, linewidth=1, linestyle='--',
                           label='Unaberrated baseline')
        ax.set_xlabel('Aberration RMS', color=ink)
        ax.set_title(label, color=ink, loc='left')
        if col == 'Jaccard Index':
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='#e5e4e0', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted)
    if single_run:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles[1:], labels[1:], frameon=False, loc='lower left')
        fig.suptitle(title, color=ink)
    else:
        fig.suptitle(title, color=ink, y=1.08)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 1.03),
                   ncol=len(summaries), frameon=False)
    note = 'Line: mean over images' + ('; band: ±1 std' if show_std else '') + \
        '. RMS 0 = unaberrated baseline. RMSE uses matched (true-positive) emitters only.'
    fig.text(0.5, -0.02, note, ha='center', color=muted, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_threshold_sweep(summary: pd.DataFrame, save_path: str, title: str, default_threshold: float):
    """Jaccard, precision and recall vs aberration RMS, one line per decode threshold."""
    ink, muted = '#0b0b0b', '#52514e'
    thresholds = sorted(summary['Threshold'].unique())
    ramp_idx = np.linspace(0, len(SEQUENTIAL_BLUES) - 1, len(thresholds)).round().astype(int)
    metrics = [('Jaccard Index', 'Jaccard index'), ('Pooled Precision', 'Precision'), ('Pooled Recall', 'Recall')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (col, label) in zip(axes, metrics):
        for threshold, idx in zip(thresholds, ramp_idx):
            rows = at_threshold(summary, threshold)
            values = rows[f'{col} mean'] if f'{col} mean' in rows else rows[col]
            is_default = np.isclose(threshold, default_threshold)
            ax.plot(rows['Aberration RMS'], values, color=SEQUENTIAL_BLUES[idx], linewidth=2.5 if is_default else 1.5,
                    marker='o', markersize=5, markeredgecolor='white', markeredgewidth=1,
                    label=f'{threshold:g}' + (' (default)' if is_default else ''))
        ax.set_xlabel('Aberration RMS', color=ink)
        ax.set_title(label, color=ink, loc='left')
        ax.set_ylim(0, 1.02)
        ax.grid(axis='y', color='#e5e4e0', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted)
    fig.suptitle(title, color=ink, y=1.08)
    fig.legend(*axes[0].get_legend_handles_labels(), title='Decode threshold', loc='upper center',
               bbox_to_anchor=(0.5, 1.03), ncol=len(thresholds), frameon=False)
    fig.text(0.5, -0.02, 'Jaccard: mean over images. Precision and recall: pooled over all emitters of a level. '
             'RMS 0 = unaberrated baseline.', ha='center', color=muted, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def evaluate_aberration_sweep(weights_path, data_root, noise=False, baseline_variant=None, save_dir=None,
                              device='cuda:0', blob_r=2.0, threshold=40, jaccard_threshold=0.1, thresholds=None):
    abr_variant = ABR_NOISY if noise else ABR_CLEAN
    baseline_variant = baseline_variant or (ORIG_NOISY if noise else ORIG_CLEAN)
    tag = 'noisy' if noise else 'clean'
    net, checkpoint_path = load_net(weights_path, device)
    if save_dir is None:
        save_dir = os.path.dirname(checkpoint_path)
    os.makedirs(save_dir, exist_ok=True)

    levels = find_levels(data_root)

    thresholds = sorted(set(thresholds or []) | {threshold})
    eval_kwargs = dict(device=device, blob_r=blob_r, thresholds=thresholds, jaccard_threshold=jaccard_threshold)
    all_results = []
    # baseline: unaberrated images of the first level, plotted at RMS = 0
    _, first_sub, localizations, param_dict = levels[0]
    df = evaluate_images(net, os.path.join(data_root, first_sub, 'x', baseline_variant), localizations,
                         param_dict, desc=f'baseline ({first_sub}/{baseline_variant})', **eval_kwargs)
    all_results.append(df.assign(Condition='baseline', Level=first_sub, **{'Aberration RMS': 0.0}))
    for rms, sub, localizations, param_dict in levels:
        df = evaluate_images(net, os.path.join(data_root, sub, 'x', abr_variant), localizations,
                             param_dict, desc=f'{sub}/{abr_variant}', **eval_kwargs)
        all_results.append(df.assign(Condition=abr_variant, Level=sub, **{'Aberration RMS': rms}))

    results_df = pd.concat(all_results, ignore_index=True)
    summary = summarize(results_df)
    results_df.to_csv(os.path.join(save_dir, f'results_per_image_{tag}.csv'), index=False)
    summary.to_csv(os.path.join(save_dir, f'results_summary_{tag}.csv'), index=False)
    plot_path = os.path.join(save_dir, f'metrics_vs_aberration_{tag}.png')
    title = f"{os.path.basename(checkpoint_path)} — {abr_variant} (baseline: {baseline_variant})"
    plot_metrics_vs_aberration({os.path.basename(checkpoint_path): at_threshold(summary, threshold)}, plot_path,
                               title=f"{title}, threshold {threshold:g}")
    if len(thresholds) > 1:
        plot_threshold_sweep(summary, os.path.join(save_dir, f'threshold_sweep_{tag}.png'), title, threshold)

    columns = ['Condition', 'Aberration RMS', 'Threshold', 'Jaccard Index mean', 'Pooled Precision', 'Pooled Recall',
               'RMSE_xy (nm) mean', 'RMSE_z (nm) mean', 'Num Empty Predictions']
    with pd.option_context('display.width', 200, 'display.max_columns', None, 'display.max_rows', None,
                           'display.precision', 4):
        print(summary[columns].sort_values(['Threshold', 'Aberration RMS']).to_string(index=False))
    print(f"Saved results and plot to {save_dir}")
    return results_df, summary


def default_run_labels(summary_paths: list) -> list:
    """Name each run by its weights folder, adding parent folders until the names are unique."""
    parts = [os.path.normpath(os.path.dirname(os.path.abspath(p))).split(os.sep) for p in summary_paths]
    for depth in range(1, max(len(p) for p in parts) + 1):
        labels = ['/'.join(p[-depth:]) for p in parts]
        if len(set(labels)) == len(labels):
            return labels
    return summary_paths


def compare_sweeps(summary_paths: list, labels: list = None, save_path: str = None, show_std: bool = True,
                   title: str = None, threshold: float = 40):
    """Overlay several saved results_summary_*.csv files on one figure (at one decode threshold)."""
    labels = labels or default_run_labels(summary_paths)
    if len(labels) != len(summary_paths):
        raise ValueError(f"Got {len(labels)} labels for {len(summary_paths)} summary files")
    summaries = {label: at_threshold(pd.read_csv(path), threshold) for label, path in zip(labels, summary_paths)}
    conditions = sorted({c for s in summaries.values() for c in s['Condition'] if c != 'baseline'})
    if save_path is None:
        common_dir = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in summary_paths])
        tags = {ABR_CLEAN: 'clean', ABR_NOISY: 'noisy'}
        tag = tags.get(conditions[0], conditions[0]) if len(conditions) == 1 else 'mixed'
        save_path = os.path.join(common_dir, f'aberration_sweep_comparison_{tag}.png')
    if title is None:
        title = f"Aberration sweep comparison — {', '.join(conditions)}"
    plot_metrics_vs_aberration(summaries, save_path, title, show_std=show_std)

    metrics = ['Jaccard Index mean', 'RMSE_xy (nm) mean', 'RMSE_z (nm) mean']
    table = pd.concat({label: s.set_index('Aberration RMS')[metrics] for label, s in summaries.items()}, axis=1)
    with pd.option_context('display.width', 200, 'display.max_columns', None, 'display.precision', 3):
        print(table)
    print(f"Saved comparison plot to {save_path}")
    return save_path


if __name__ == "__main__":
    args = get_args()
    if args.compare:
        compare_sweeps(args.compare, args.labels, args.save_path, show_std=not args.no_std, threshold=args.threshold)
    else:
        evaluate_aberration_sweep(args.weights_path, args.data_root, args.noise, args.baseline_variant, args.save_dir,
                                  args.device, args.blob_r, args.threshold, args.jaccard_threshold, args.thresholds)

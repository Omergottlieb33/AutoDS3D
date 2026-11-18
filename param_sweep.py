"""
Sweep of the label/volume parameters: number of z-voxels (D), blob sigma and blob radius.

The simulated images do not depend on these parameters, only the labels do. So the images are
simulated once (train + test, with continuous ground-truth positions), every configuration gets
its own labels built from the same images, and all trained models are scored on the same test set
against the continuous positions. Differences between configurations come from the labels alone.

Stages (driven by run_param_sweep.sh):
    images     simulate images + continuous GT positions from a base param.pickle
    labels     build y.pickle / param.pickle of one configuration for an image set
    evaluate   decode the test set with a trained model at several thresholds
    summarize  collect all configurations into one table and plots
"""
import os
import sys
import json
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from skimage import io
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from DS3Dplus.ds3d_utils import ImModelTraining, Volume2XYZ


class _CompatUnpickler(pickle.Unpickler):
    """Reads pickles written with numpy 2.x on numpy 1.x too."""
    def find_class(self, module, name):
        if module.startswith('numpy._core') and np.__version__.startswith('1.'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, 'rb') as f:
        return _CompatUnpickler(f).load()


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def set_geometry(p, D):
    """Voxel grid of the reconstruction volume, as func_utils.prepare_training_params sets it."""
    p['D'] = int(D)
    p['vs_z'] = (p['zrange'][1] - p['zrange'][0]) / p['D']
    return p


# ----------------------------------------------------------------------------------------- images
def sample_emitters(p, rng):
    """Continuous positions over the same region and photon/emitter-count ranges as Sampling."""
    n = rng.integers(p['num_particles_range'][0], p['num_particles_range'][1] + 1)
    # Sampling places centres on voxels [buffer, size-buffer) +-0.49 voxel; here the same span, continuous
    x_vox = rng.uniform(p['buffer_WW'] - 0.5, p['WW'] - p['buffer_WW'] - 0.5, n)
    y_vox = rng.uniform(p['buffer_HH'] - 0.5, p['HH'] - p['buffer_HH'] - 0.5, n)
    x = (x_vox - (p['WW'] - 1) / 2) * p['vs_xy']
    y = (y_vox - (p['HH'] - 1) / 2) * p['vs_xy']
    z = rng.uniform(p['zrange'][0], p['zrange'][1], n)
    photons = rng.integers(p['Nsig_range'][0], p['Nsig_range'][1], n)
    return np.c_[x, y, z, photons].astype(np.float64)


def cmd_images(args):
    p = load_pickle(args.base_params)
    p['device'] = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    x_folder = os.path.join(args.out, 'x')
    os.makedirs(x_folder, exist_ok=True)
    model = ImModelTraining(p)
    gt = {}
    for i in range(args.n):
        xyzps = sample_emitters(p, rng)
        with torch.no_grad():
            im = model(torch.from_numpy(xyzps).to(p['device'])).cpu().numpy().astype(np.uint16)
        name = f'{i:05d}.tif'
        io.imsave(os.path.join(x_folder, name), im, check_contrast=False)
        gt[name] = xyzps
        if (i + 1) % max(args.n // 10, 1) == 0:
            print(f'image [{i + 1} / {args.n}]', flush=True)
    save_pickle(gt, os.path.join(args.out, 'gt.pickle'))
    save_pickle(p, os.path.join(args.out, 'base_param.pickle'))


# ----------------------------------------------------------------------------------------- labels
def make_label(xyzps, p, blob_sigma, blob_r, normal_factor):
    """Voxel ids + 3D Gaussian blobs of one image, the inverse of Sampling.xyzp_batch's mapping."""
    x_vox = xyzps[:, 0] / p['vs_xy'] + (p['WW'] - 1) / 2
    y_vox = xyzps[:, 1] / p['vs_xy'] + (p['HH'] - 1) / 2
    z_vox = (xyzps[:, 2] - p['zrange'][0]) / p['vs_z'] - 0.5
    vox = np.c_[x_vox, y_vox, z_vox]
    ids = np.rint(vox).astype(int)
    ids[:, 2] = np.clip(ids[:, 2], 0, p['D'] - 1)
    local = vox - ids

    xs = np.arange(-blob_r, blob_r + 1, dtype=np.float64)
    zz, yy, xx = np.meshgrid(xs, xs, xs, indexing='ij')
    blob3d = np.exp(-0.5 * ((xx - local[:, 0, None, None, None]) ** 2 +
                            (yy - local[:, 1, None, None, None]) ** 2 +
                            (zz - local[:, 2, None, None, None]) ** 2) / blob_sigma ** 2)
    blob3d = blob3d * normal_factor * xyzps[:, 3][:, None, None, None]
    return ids, blob3d


def cmd_labels(args):
    gt = load_pickle(os.path.join(args.images, 'gt.pickle'))
    p = load_pickle(os.path.join(args.images, 'base_param.pickle'))
    p = set_geometry(p, args.D)
    p['blob_sigma'] = args.blob_sigma
    p['blob_r'] = args.blob_r
    p['td_folder'] = args.out
    p['n_ims'] = len(gt)
    normal_factor = p['blob_maxv'] / p['Nsig_range'][1]  # as Sampling.normal_factor2

    labels = {'volume_size': (p['D'], p['HH'], p['WW']), 'us_factor': p['us_factor'],
              'blob_r': args.blob_r, 'blob_maxv': p['blob_maxv']}
    for name, xyzps in gt.items():
        ids, blob3d = make_label(xyzps, p, args.blob_sigma, args.blob_r, normal_factor)
        labels[name] = {'xyz_ids': ids, 'blob3d': blob3d.astype(np.float32), 'xyzps': xyzps}

    os.makedirs(args.out, exist_ok=True)
    save_pickle(labels, os.path.join(args.out, 'y.pickle'))
    save_pickle(p, os.path.join(args.out, 'param.pickle'))
    print(f'labels: D={p["D"]} (vs_z {p["vs_z"] * 1e3:.1f} nm), blob_sigma={args.blob_sigma}, '
          f'blob_r={args.blob_r} -> {args.out}')


# --------------------------------------------------------------------------------------- evaluate
def load_net(run_dir, device):
    pts = sorted(glob.glob(os.path.join(run_dir, 'net_*.pt')), key=os.path.getmtime)
    if not pts:
        raise FileNotFoundError(f'no net_*.pt in {run_dir}')
    ckpt = torch.load(pts[-1], map_location=device, weights_only=False)
    net = ckpt['net']
    net.load_state_dict(ckpt['state_dict'])
    return net.to(device).eval(), pts[-1]


def match(xyz_gt, xyz_pred, radius):
    """Hungarian matching; returns (gt index, pred index) of pairs closer than radius."""
    if xyz_pred is None or len(xyz_pred) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    C = cdist(xyz_gt, xyz_pred)
    gi, pi = linear_sum_assignment(C)
    keep = C[gi, pi] < radius
    return gi[keep], pi[keep]


def score(rows):
    """Pooled metrics of a list of per-image rows."""
    tp = sum(r['tp'] for r in rows)
    n_gt = sum(r['n_gt'] for r in rows)
    n_pred = sum(r['n_pred'] for r in rows)
    err = np.concatenate([r['err'] for r in rows]) if tp else np.zeros((0, 3))
    out = {'jaccard': tp / (n_gt + n_pred - tp) if n_gt + n_pred else 0.0,
           'recall': tp / n_gt if n_gt else 0.0,
           'precision': tp / n_pred if n_pred else 0.0}
    if tp:
        out['rmse_xy_nm'] = float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, 1)))) * 1e3
        out['rmse_z_nm'] = float(np.sqrt(np.mean(err[:, 2] ** 2))) * 1e3
        out['bias_z_nm'] = float(np.mean(err[:, 2])) * 1e3
    return out


def z_locking(z_pred, p, bins=10):
    """Non-uniformity of the sub-voxel z phase of the localizations: 0 = flat, larger = locked to voxels."""
    phase = np.mod((z_pred - p['zrange'][0]) / p['vs_z'], 1.0)
    hist, _ = np.histogram(phase, bins=bins, range=(0, 1))
    hist = hist / max(hist.mean(), 1e-9)
    return float(hist.std())


def cmd_evaluate(args):
    device = torch.device(args.device)
    p = load_pickle(os.path.join(args.config, 'param.pickle'))
    gt = load_pickle(os.path.join(args.test, 'gt.pickle'))
    net, ckpt = load_net(args.run, device)
    thresholds = [float(t) for t in args.thresholds.split(',')]
    decoders = {t: Volume2XYZ(dict(p, device=device, threshold=t)) for t in thresholds}

    names = sorted(gt)
    rows = {t: [] for t in thresholds}
    for name in names:
        im = io.imread(os.path.join(args.test, 'x', name)).astype(np.float32)
        if p.get('project_01'):
            im = (im - im.min()) / (im.max() - im.min())
        with torch.no_grad():
            vol = net(torch.from_numpy(im)[None, None].to(device))
        xyz_gt = gt[name][:, :3]
        for t, decode in decoders.items():
            xyz_pred, _ = decode(vol)
            gi, pi = match(xyz_gt, xyz_pred, args.radius)
            rows[t].append({'n_gt': len(xyz_gt), 'n_pred': 0 if xyz_pred is None else len(xyz_pred),
                            'tp': len(gi), 'err': (xyz_pred[pi] - xyz_gt[gi]) if len(gi) else np.zeros((0, 3)),
                            'z_pred': xyz_pred[pi, 2] if len(gi) else np.zeros(0)})

    # pick the threshold on the first half of the test set, report on the second half
    half = len(names) // 2
    per_t = {t: {'select': score(r[:half]), 'report': score(r[half:])} for t, r in rows.items()}
    best_t = max(thresholds, key=lambda t: per_t[t]['select']['jaccard'])
    rep = rows[best_t][half:]
    z_pred = np.concatenate([r['z_pred'] for r in rep])

    result = {'config': os.path.basename(os.path.normpath(args.config)), 'checkpoint': ckpt,
              'D': p['D'], 'vs_z_nm': p['vs_z'] * 1e3, 'blob_sigma': p['blob_sigma'], 'blob_r': p['blob_r'],
              'blob_sigma_z_nm': p['blob_sigma'] * p['vs_z'] * 1e3,
              'blob_sigma_xy_nm': p['blob_sigma'] * p['vs_xy'] * 1e3,
              'radius_nm': args.radius * 1e3, 'threshold': best_t, 'n_test_images': len(names) - half,
              **per_t[best_t]['report'], 'z_locking': z_locking(z_pred, p),
              'per_threshold': {str(t): v['report'] for t, v in per_t.items()}}

    # z error vs depth, to see whether a configuration fails at the ends of the range
    err = np.concatenate([r['err'] for r in rep]) if len(z_pred) else np.zeros((0, 3))
    z_true = z_pred - err[:, 2]
    edges = np.linspace(p['zrange'][0], p['zrange'][1], 11)
    idx = np.clip(np.digitize(z_true, edges) - 1, 0, 9)
    result['rmse_z_by_depth_nm'] = [float(np.sqrt(np.mean(err[idx == k, 2] ** 2)) * 1e3) if np.any(idx == k)
                                    else None for k in range(10)]
    result['depth_bin_edges_um'] = edges.tolist()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_threshold', 'rmse_z_by_depth_nm',
                                                                     'depth_bin_edges_um')}, indent=2))


# -------------------------------------------------------------------------------------- summarize
def cmd_summarize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(os.path.join(args.root, 'eval', '*.json')))
    if not files:
        sys.exit(f'no results in {os.path.join(args.root, "eval")}')
    results = [json.load(open(f)) for f in files]
    cols = ['config', 'D', 'vs_z_nm', 'blob_sigma', 'blob_r', 'blob_sigma_z_nm', 'blob_sigma_xy_nm',
            'threshold', 'jaccard', 'recall', 'precision', 'rmse_xy_nm', 'rmse_z_nm', 'bias_z_nm', 'z_locking']
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in results]).sort_values(['blob_sigma', 'blob_r', 'D'])
    df.to_csv(os.path.join(args.root, 'summary.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(df.round(3).to_string(index=False))

    metrics = [('jaccard', 'Jaccard (higher better)'), ('rmse_z_nm', 'RMSE z [nm]'),
               ('rmse_xy_nm', 'RMSE xy [nm]'), ('z_locking', 'z voxel locking (0 = none)')]
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.2 * len(metrics), 7.5))
    # row 1: vs D, one line per blob (sigma, r); row 2: vs blob sigma, one line per D
    for (key, label), ax in zip(metrics, axes[0]):
        for (s, r), g in df.groupby(['blob_sigma', 'blob_r']):
            if len(g) > 1:
                ax.plot(g['D'], g[key], 'o-', label=f'sigma {s}, r {r}')
        ax.set_xlabel('D (z voxels)'); ax.set_title(label); ax.grid(alpha=.3)
    for (key, label), ax in zip(metrics, axes[1]):
        for D, g in df.groupby('D'):
            if len(g) > 1:
                g = g.sort_values('blob_sigma')
                ax.plot(g['blob_sigma'], g[key], 'o-', label=f'D {D}')
        ax.set_xlabel('blob sigma [voxels]'); ax.set_title(label); ax.grid(alpha=.3)
    for row in axes:
        if row[0].get_legend_handles_labels()[0]:
            row[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.root, 'summary.png'), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        e = r['depth_bin_edges_um']
        ax.plot([(a + b) / 2 for a, b in zip(e[:-1], e[1:])],
                [np.nan if v is None else v for v in r['rmse_z_by_depth_nm']], 'o-', ms=3, label=r['config'])
    ax.set_xlabel('z [um]'); ax.set_ylabel('RMSE z [nm]'); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.root, 'rmse_z_by_depth.png'), dpi=150)
    print(f'\nwrote {os.path.join(args.root, "summary.csv")}, summary.png, rmse_z_by_depth.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('images')
    a.add_argument('--base_params', required=True, help='param.pickle of the experiment (optics, noise, photons)')
    a.add_argument('--out', required=True)
    a.add_argument('--n', type=int, required=True)
    a.add_argument('--seed', type=int, required=True)
    a.add_argument('--device', default='cuda:0')

    a = sub.add_parser('labels')
    a.add_argument('--images', required=True, help='folder written by the images stage')
    a.add_argument('--out', required=True)
    a.add_argument('--D', type=int, required=True)
    a.add_argument('--blob_sigma', type=float, required=True)
    a.add_argument('--blob_r', type=int, required=True)

    a = sub.add_parser('evaluate')
    a.add_argument('--config', required=True, help='folder written by the labels stage')
    a.add_argument('--run', required=True, help='folder with the trained net_*.pt')
    a.add_argument('--test', required=True, help='test folder written by the images stage')
    a.add_argument('--out', required=True, help='result .json')
    a.add_argument('--thresholds', default='10,20,40,60,80,120,160,200,300')
    a.add_argument('--radius', type=float, default=0.1, help='matching radius [um]')
    a.add_argument('--device', default='cuda:0')

    a = sub.add_parser('summarize')
    a.add_argument('--root', required=True)

    args = ap.parse_args()
    {'images': cmd_images, 'labels': cmd_labels, 'evaluate': cmd_evaluate, 'summarize': cmd_summarize}[args.cmd](args)


if __name__ == '__main__':
    main()

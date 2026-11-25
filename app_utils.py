
import numpy as np
from numpy import pi
from skimage import io
import tifffile
import csv
import time
import os
import shutil
import pickle
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import torch
import torch.fft as fft
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DS3Dplus.ds3d_utils import ImModel, Sampling, MyDataset, Volume2XYZ, calc_jaccard_rmse, KDE_loss3D, ImModelBead, ImModelBase, ImModelTraining
from DS3Dplus.ds3d_utils import LON as Net
from DS3Dplus.training_utils import TorchTrainer
import matplotlib.pyplot as plt
from tqdm import tqdm


def out_path(param_dict, file_name):
    """Path of a product file inside the user-selected output folder."""
    return os.path.join(param_dict.get('output_dir', os.getcwd()), file_name)



class ImModel_pr(nn.Module):
    def __init__(self, params):
        """
        a scalar model for air or oil objective in microscopy
        """
        super().__init__()
        device = params['device']
        ################### set parameters: unit:um
        # oil objective
        M = params['M']  # magnification
        NA = params['NA']  # NA
        n_immersion = params['n_immersion']  # refractive index of the immersion of the objective
        lamda = params['lamda']  # wavelength
        n_sample = params['n_sample']  # refractive index of the sample
        f_4f = params['f_4f']  # focal length of 4f system
        ps_camera = params['ps_camera']  # pixel size of the camera
        ps_BFP = params['ps_BFP']  # pixel size at back focal plane

        # image
        H, W = params['H'], params['W']  # FOV size
        g_size = 9  # size of the gaussian blur kernel
        g_sigma = params['g_sigma']  # std of the gaussian blur kernel

        ###################

        N = np.floor(f_4f * lamda / (ps_camera * ps_BFP))  # simulation size
        N = int(N + 1 - (N % 2))  # make it odd
        print(f'Simulation size of the imaging model is {N} which must be larger than image size (PSF z-stack and training images)!')

        # pupil/aperture at back focal plane
        d_pupil = 2 * f_4f * NA / np.sqrt(M ** 2 - NA ** 2)  # diameter [um]
        pn_pupil = d_pupil / ps_BFP  # pixel number of the pupil diameter should be smaller than the simulation size N
        if N < pn_pupil:
            raise Exception('Simulation size is smaller than the pupil!')
        # cartesian and polar grid in BFP
        x_phys = np.linspace(-N / 2, N / 2, N) * ps_BFP
        xi, eta = np.meshgrid(x_phys, x_phys)  # cartesian physical coordinates
        r_phys = np.sqrt(xi ** 2 + eta ** 2)
        pupil = (r_phys < d_pupil / 2).astype(np.float32)

        x_ang = np.linspace(-1, 1, N) * (N / pn_pupil) * (NA / n_immersion)  # angular coordinate
        xx_ang, yy_ang = np.meshgrid(x_ang, x_ang)
        r = np.sqrt(
            xx_ang ** 2 + yy_ang ** 2)  # normalized angular coordinates, s.t. r = NA/n_immersion at edge of E field support

        k_immersion = 2 * pi * n_immersion / lamda  # [1/um]
        sin_theta_immersion = r
        circ_NA = (sin_theta_immersion < (NA / n_immersion)).astype(
            np.float32)  # the same as pupil, NA / n_immersion < 1
        cos_theta_immersion = np.sqrt(1 - (sin_theta_immersion * circ_NA) ** 2) * circ_NA

        k_sample = 2 * pi * n_sample / lamda
        sin_theta_sample = n_immersion / n_sample * sin_theta_immersion
        # note: when circ_sample is smaller than circ_NA, super angle fluorescence apears
        circ_sample = (sin_theta_sample < 1).astype(np.float32)  # if all the frequency of the sample can be captured
        cos_theta_sample = np.sqrt(1 - (sin_theta_sample * circ_sample) ** 2) * circ_sample * circ_NA

        # circular aperture to impose on BFP, SAF is excluded
        circ = circ_NA * circ_sample

        pn_circ = np.floor(np.sqrt(np.sum(circ) / pi) * 2)
        pn_circ = int(pn_circ + 1 - (pn_circ % 2))
        Xgrid = 2 * pi * xi * M / (lamda * f_4f)
        Ygrid = 2 * pi * eta * M / (lamda * f_4f)
        Zgrid = k_sample * cos_theta_sample
        NFPgrid = k_immersion * (-1) * cos_theta_immersion  # -1

        self.device = device
        self.Xgrid = torch.from_numpy(Xgrid).to(device)
        self.Ygrid = torch.from_numpy(Ygrid).to(device)
        self.Zgrid = torch.from_numpy(Zgrid).to(device)
        self.NFPgrid = torch.from_numpy(NFPgrid).to(device)
        self.circ = torch.from_numpy(circ).to(device)
        self.circ_NA = torch.from_numpy(circ_NA).to(device)
        self.circ_sample = torch.from_numpy(circ_sample).to(device)
        self.idx05 = int(N / 2)
        self.N = N
        self.pn_pupil = pn_pupil
        self.pn_circ = pn_circ

        # for a blur kernel
        g_r = int(g_size / 2)
        g_xs = torch.linspace(-g_r, g_r, g_size, device=device).type(torch.float64)
        self.g_xx, self.g_yy = torch.meshgrid(g_xs, g_xs, indexing='xy')

        # crop settings
        self.r0, self.c0 = int(np.round((N-H)/2)), int(np.round((N-W)/2))
        # h05, w05 = int(H / 2), int(W / 2)
        # self.h05, self.w05 = h05, w05
        self.H, self.W = H, W

        self.phase_mask = torch.tensor(circ, device=device, requires_grad=True)
        self.g_sigma = torch.tensor(g_sigma, device=device, requires_grad=True)

    def forward(self, xyzps, NFPs):
        phase_lateral = self.Xgrid * (xyzps[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzps[:, 1:2].unsqueeze(1))
        phase_axial = self.Zgrid * (xyzps[:, 2:3].unsqueeze(1)) + self.NFPgrid * NFPs.unsqueeze(1).unsqueeze(1)
        ef_bfp = self.circ * torch.exp(1j * (phase_axial + phase_lateral + self.phase_mask))
        psf_field = fft.fftshift(fft.fftn(fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))  # FT
        psfs = torch.abs(psf_field) ** 2

        # blur
        blur_kernel = 1 / (2 * pi * self.g_sigma ** 2) * (
            torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / self.g_sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        psfs = psfs.squeeze(1)

        # photon normalization
        psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)  # photon normalization
        # psfs = psfs[:, self.idx05 - self.h05:self.idx05 + self.h05 + 1, self.idx05 - self.w05:self.idx05 + self.w05 + 1]
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

        return psfs


def calculate_cc(output, target):
    # output: rank 3, target: rank 3
    output_mean = np.mean(output, axis=(1, 2), keepdims=True)
    target_mean = np.mean(target, axis=(1, 2), keepdims=True)
    ccs = (np.sum((output - output_mean) * (target - target_mean), axis=(1, 2)) /
           (np.sqrt(np.sum((output - output_mean) ** 2, axis=(1, 2)) * np.sum((target - target_mean) ** 2,
                                                                              axis=(1, 2))) + 1e-9))
    return ccs


def phase_retrieval(param_dict, pr_dict, fig_flag=True):

    device = param_dict['device']

    file_path = pr_dict['zstack_file_path']

    nfps = pr_dict['nfps']
    r_bead = pr_dict['r_bead']
    epoch_num = pr_dict['epoch_num']
    loss_label = pr_dict['loss_label']

    # read the zstack and set the image size for the imaging model
    zstack = io.imread(file_path)  # axis0 -- z position

    corner_size = max(7, int(0.1*zstack.shape[1]))

    param_dict['H'], param_dict['W'] = zstack.shape[1], zstack.shape[2]

    # estimate Gaussian noise: mean and std
    patches = np.concatenate(
        (np.concatenate((zstack[:, :corner_size, :corner_size], zstack[:, :corner_size, -corner_size:]), axis=2),
         np.concatenate((zstack[:, -corner_size:, :corner_size], zstack[:, -corner_size:, -corner_size:]), axis=2)),
        axis=1)
    means = np.mean(patches, axis=(1, 2), keepdims=True)
    stds = np.std(patches, axis=(1, 2), keepdims=True)
    # filtering mask
    zstack = zstack - means
    mask = (zstack > stds)
    # erode and dimate the mask
    struct = ndimage.generate_binary_structure(2, 1)  # raius 1 or 2
    mask = [ndimage.binary_dilation(ndimage.binary_erosion(mask[i, :, :], struct), struct) for i in range(mask.shape[0])]
    # clean zstack
    zstack = zstack * np.array(mask)
    z_photons = np.sum(zstack, axis=(1, 2))

    im_model_bead = ImModelBead(param_dict)

    print(f'BFP aperture in pixel unit: {int(np.round(im_model_bead.pn_pupil))}/{im_model_bead.N}.')

    im_model_bead.phase_mask.requires_grad_(True)
    im_model_bead.g_sigma.requires_grad_(True)

    num_zs = zstack.shape[0]
    xyzps = np.zeros((num_zs, 4))
    xyzps[:, 3] = z_photons
    xyzps = torch.tensor(xyzps, device=device)
    nfps_np = nfps.copy()
    nfps = torch.tensor(nfps, device=device).unsqueeze(1)

    y = torch.tensor(zstack, device=device)  # measurement
    optimizer = torch.optim.Adam([{'params': im_model_bead.phase_mask, 'lr': 0.1},
                                  {'params': im_model_bead.g_sigma, 'lr': 0.06}
                                  ])
    epoch_loss = []
    for i in range(100):
        fx = im_model_bead(xyzps, nfps)

        loss = torch.nn.functional.mse_loss(fx, y)  # mse
        # loss = torch.mean(fx-y*torch.log(fx))  # gauss log likelihood

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    optimizer = torch.optim.Adam([{'params': im_model_bead.phase_mask, 'lr': 0.02},
                                  {'params': im_model_bead.g_sigma, 'lr': 0.01}
                                  ])
    for i in range(100):
        with torch.no_grad():
            fx = im_model_bead(xyzps, nfps)
            model_psfs = fx.detach().cpu().numpy()
            ccs = calculate_cc(zstack, model_psfs)
            ids = np.argsort(ccs)[:5]

        fx = im_model_bead(xyzps[ids], nfps[ids])
        loss = torch.nn.functional.mse_loss(fx, y[ids])
        # loss = torch.mean(fx-y[ids]*torch.log(fx))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    with torch.no_grad():
        fx = im_model_bead(xyzps, nfps)
        model_psfs = fx.detach().cpu().numpy()
        ccs = calculate_cc(zstack, model_psfs)

    mask_rec = im_model_bead.phase_mask.detach().cpu().numpy()
    mask_rec = np.angle(np.exp(1j * mask_rec))
    g_sigma = im_model_bead.g_sigma.detach().item()
    psfs_np = model_psfs
    phase_mask = mask_rec


    if fig_flag:
        fig = plt.figure(1, figsize=(7, 4))
        gs = fig.add_gridspec(3, 7)

        ax = fig.add_subplot(gs[:2, :3])
        maskplot = ax.imshow(phase_mask)
        plt.colorbar(maskplot)
        ax.set_title('retrieved phase')

        ax = fig.add_subplot(gs[0, 3:])
        ids = (0, (zstack.shape[0]-1) // 4, (zstack.shape[0]-1) // 2, ((zstack.shape[0]-1)//4)*3, -1)
        im_demo = np.concatenate((zstack[ids[0]], zstack[ids[1]], zstack[ids[2]], zstack[ids[3]],
                                  zstack[ids[4]]), axis=1)
        ax.imshow(im_demo)
        ax.axis('off')
        ax.set_title('exp')

        ax = fig.add_subplot(gs[1, 3:])
        im_demo = np.concatenate((psfs_np[ids[0]], psfs_np[ids[1]], psfs_np[ids[2]], psfs_np[ids[3]],
                                  psfs_np[ids[4]]), axis=1)
        ax.imshow(im_demo)
        ax.axis('off')
        ax.set_title('model')

        ax = fig.add_subplot(gs[2, :])
        ax.plot(nfps_np, ccs)
        ax.set_xlabel('NFP [um]')
        ax.set_ylabel('CC')
        # ax.set_title('model accuracy')

        plt.savefig(out_path(param_dict, 'phase_retrieval_results.jpg'), bbox_inches='tight', dpi=300)
        plt.clf()
        # print(f'phase retrieval results: phase_retrieval_results.jpg')

    return phase_mask, g_sigma, ccs


def show_z_psf(param_dict):
    model = ImModel(param_dict)
    model.model_demo(np.linspace(param_dict['zrange'][0], param_dict['zrange'][1], 5))  # check PSFs


IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def list_image_files(folder):
    return sorted(n for n in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, n)) and os.path.splitext(n)[1].lower() in IMAGE_EXTS)


def read_stack(path):
    """Read an image file as a stack (n_frames, H, W). A single image gives n_frames=1."""
    stack = tifffile.imread(path) if path.lower().endswith(('.tif', '.tiff')) else io.imread(path)
    return stack[np.newaxis] if stack.ndim == 2 else stack


def frame_counts(folder):
    """[(file name, number of frames)] for a folder of single- or multi-frame images (sorted by name)."""
    counts = []
    for name in list_image_files(folder):
        path = os.path.join(folder, name)
        if name.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(path) as tf:
                shape = tf.series[0].shape
            counts.append((name, shape[0] if len(shape) == 3 else 1))
        else:
            counts.append((name, 1))
    return counts


def iter_frames(folder):
    """Yield (global 0-based frame index, 2D frame) over all files of the folder, reading one file at a time."""
    idx = 0
    for name, _ in frame_counts(folder):
        for frame in read_stack(os.path.join(folder, name)):
            yield idx, frame
            idx += 1


def read_frame(folder, frame_idx):
    """Read one frame by its global 0-based index without loading the other files."""
    start = 0
    for name, n in frame_counts(folder):
        if frame_idx < start + n:
            return read_stack(os.path.join(folder, name))[frame_idx - start]
        start += n
    raise IndexError(f'frame {frame_idx} is out of range ({start} frames in {folder})')


def read_last_frames(folder, num):
    """The last `num` frames (or all, if fewer) as an array (n, H, W).

    If the folder holds multi-frame tifs, only the last file is used (its last `num` frames);
    for single-frame images the last `num` files are used.
    """
    counts = frame_counts(folder)
    if any(n > 1 for _, n in counts):
        counts = counts[-1:]
    frames, need = [], num
    for name, n in reversed(counts):
        if need <= 0:
            break
        stack = read_stack(os.path.join(folder, name))
        frames.insert(0, stack[-need:])
        need -= len(stack[-need:])
    return np.concatenate(frames, axis=0)


def background_removal(im_folder, save_folder=None, num=100):
    """Subtract the per-pixel minimum of every `num` consecutive frames.

    Each input file (a single image or a multi-frame tif) is decomposed into frames and written
    back as one multi-frame tif named <name>_br.tif in save_folder.
    """
    if save_folder is None:
        save_folder = im_folder + '_br'  # where to save the images after background removal
    os.makedirs(os.path.dirname(os.path.normpath(save_folder)), exist_ok=True)

    if os.path.exists(save_folder) and len(os.listdir(save_folder)) > 0:
        print('probably has been done!')
        return save_folder
    if num <= 0:
        raise ValueError('num must be a positive integer.')
    im_files = list_image_files(im_folder)
    if len(im_files) == 0:
        raise ValueError(f'No supported image files found in: {im_folder}')
    os.makedirs(save_folder, exist_ok=True)

    for name in tqdm(im_files, desc='Background removal'):
        stack = read_stack(os.path.join(im_folder, name))
        out = np.empty_like(stack)
        for start in range(0, len(stack), num):  # chunks of `num` frames, minimum map per chunk
            chunk = stack[start:start + num]
            out[start:start + num] = chunk - np.min(chunk, axis=0)
        tifffile.imwrite(os.path.join(save_folder, os.path.splitext(name)[0] + '_br.tif'), out)

    return save_folder

def iter_sampled_frames(folder, n=6000):
    """Yield (global frame indices, frames) per file for about n frames spread evenly over the folder."""
    counts = frame_counts(folder)
    total = sum(c for _, c in counts)
    wanted = np.unique(np.linspace(0, total - 1, min(n, total)).astype(int))
    start = 0
    for name, c in counts:
        local = wanted[(wanted >= start) & (wanted < start + c)] - start
        start += c
        if len(local) == 0:
            continue
        path = os.path.join(folder, name)
        if name.lower().endswith(('.tif', '.tiff')) and c > 1:
            sel = tifffile.imread(path, key=local.tolist())  # reads only these pages
            sel = sel[np.newaxis] if sel.ndim == 2 else sel
        else:
            sel = read_stack(path)[local]
        yield local + start - c, sel.astype(np.float32)


def blob_peaks(im, thr, size=9, det=None):
    """Values of im at the local maxima (size x size window) of det above thr; det defaults to im."""
    det = im if det is None else det
    return im[(ndimage.maximum_filter(det, size=size) == det) & (det > thr)]


def smooth3(x):
    """3x3 mean filter over the last two axes: lowers the noise ~3x, keeps the emitter peaks."""
    return ndimage.uniform_filter(x, size=(1,) * (x.ndim - 2) + (3, 3))


def find_disturbances(frame_idx, frame_max, margin, rule='max'):
    """Disturbances (e.g. room light) in a movie, from one statistic per frame.

    rule='max': frame_max is the frame maximum; frames above 3x the typical max of frames with emitters.
    rule='level': frame_max is a high percentile of the frame above its background (emitters hardly move
    it, a light leak does); frames more than 6 robust sigma above the typical value.
    Returns ([(first, last)] frame ranges including +- margin frames, ceiling, typical value)."""
    m0 = float(np.median(frame_max))  # typical value (noise only)
    s0 = float(1.4826 * np.median(np.abs(frame_max - m0)))
    if rule == 'level':
        ceiling = m0 + 6 * max(s0, 0.5)
    else:
        sig = frame_max > m0 + 6 * max(s0, 1.0)
        ceiling = 3 * float(np.median(frame_max[sig])) if sig.any() else np.inf
    jumps = np.sort(frame_idx[frame_max > ceiling])
    runs = np.split(jumps, np.where(np.diff(jumps) > 2 * margin)[0] + 1) if len(jumps) else []
    return [(int(r[0]) - margin, int(r[-1]) + margin) for r in runs], ceiling, m0


def in_ranges(idx, ranges):
    """Boolean mask: which of the frame indices idx fall in any of the (first, last) ranges."""
    idx = np.asarray(idx)
    mask = np.zeros(idx.shape, bool)
    for a, b in ranges:
        mask |= (idx >= a) & (idx <= b)
    return mask


def estimate_background(folder, n_frames=1300, disturbance_margin=150, min_frames=20, block_rows=32, verbose=True):
    """Empirical background of a raw (not background-removed) movie, one map per file (local in time).

    For each file, from frames spread over it: the per-pixel temporal median and robust temporal std
    (1.4826 MAD), in ADU. The background drifts slowly and unevenly over a long movie, so maps of the
    whole movie would blur it. Disturbance frames are left out (frames whose 99th percentile above the
    background jumps). Returns dict(bg_mean_map (n_files, H, W), bg_std_map (n_files, H, W),
    bg_offsets (frame level minus its map), disturbances). Emitters are sparse in time, so the median ignores them.
    """
    def maps(stack):  # in row blocks, to keep memory low
        med = np.empty(stack.shape[1:], np.float32)
        sd = np.empty(stack.shape[1:], np.float32)
        for r in range(0, stack.shape[1], block_rows):
            blk = stack[:, r:r + block_rows].astype(np.float32)
            m = np.median(blk, axis=0)
            med[r:r + block_rows] = m
            sd[r:r + block_rows] = 1.4826 * np.median(np.abs(blk - m), axis=0)
        return med, sd

    chunks, idx, p99 = [], [], []
    for g, c in iter_sampled_frames(folder, n_frames):
        c = c.astype(np.uint16)
        m, _ = maps(c)  # provisional map of this file
        res = c.astype(np.float32) - m
        res -= np.median(res.reshape(len(res), -1), axis=1)[:, None, None]
        chunks.append((g, c))
        idx.append(g)
        p99.append(np.percentile(res.reshape(len(res), -1), 99, axis=1))
    disturbances, _, _ = find_disturbances(np.concatenate(idx), np.concatenate(p99), disturbance_margin, rule='level')

    mean_maps, std_maps, offsets = [], [], []
    for g, c in chunks:
        keep = ~in_ranges(g, disturbances)
        if keep.sum() < min_frames:
            continue
        m, sd = maps(c[keep])
        mean_maps.append(m)
        std_maps.append(sd)
        offsets += [np.median(f.astype(np.float32) - m) for f in c[keep]]
    mean_maps, std_maps, offsets = np.array(mean_maps), np.array(std_maps), np.array(offsets, np.float32)
    if verbose:
        print(f'Background: {len(mean_maps)} maps (one per file) from {sum(len(g) for g, _ in chunks)} frames, '
              f'left out disturbances {disturbances}; mean {mean_maps.min():.0f}-{mean_maps.max():.0f} '
              f'(median {np.median(mean_maps):.0f}), temporal std {np.percentile(std_maps, 1):.1f}-'
              f'{np.percentile(std_maps, 99):.1f}, frame offsets {np.percentile(offsets, 1):.1f} to {np.percentile(offsets, 99):.1f}')
    return dict(bg_mean_map=mean_maps, bg_std_map=std_maps, bg_offsets=offsets, disturbances=disturbances)


def calibrate_nsig(param_dict, n_exp_frames=6000, n_sim_images=100, max_peaks_per_frame=60, disturbance_margin=150,
                   fit_noise=True, save_dir=None, verbose=True):
    """Find the photon range Nsig_range whose simulated emitter peaks match the experiment.

    Emitter peaks (local maxima of the 3x3-smoothed image above the noise) are collected from
    background-removed experimental frames and from simulated images (same PSF, blur, noise and
    density settings as the training data). Photons only scale the emitter signal, so the sample
    images are built once (per-emitter blurred PSFs and noise are fixed) and every candidate range
    is re-evaluated cheaply; the range is found by minimizing the distance between the peak
    quantiles of the simulation and the experiment. The result is verified with the full imaging model.

    A frame contributes at most max_peaks_per_frame (random) peaks, and peaks far above the bulk
    (> 5x the median: aggregates, saturation) are ignored. Disturbances are left out: frames whose
    max pixel jumps above 3x the typical max of frames with emitters, +- disturbance_margin frames.
    With fit_noise, param_dict['baseline'] and ['read_std'] are first set from the mean and std of the
    experimental frames without emitters (more faithful than a single background pixel).
    Raw movies: if param_dict has an empirical background (bg_mean_map, see estimate_background), the
    frames are compared after subtracting their file's temporal median and their own level, the
    simulation uses the maps as background, and fit_noise is not used. param_dict needs the training settings (see prepare_training_params),
    'im_br_folder' and an initial 'Nsig_range'. Returns (Nsig_range, info); Nsig_range is None when
    the experiment shows too few emitters to calibrate.
    """
    from scipy.optimize import minimize
    device = param_dict['device']

    # experiment: noise level from the first chunk, then emitter peaks over the whole sequence
    cand, frame_idx, frame_max, thr = [], [], [], None  # cand: (global frame index, frame, peaks)
    noise_frames = []  # (global frame index, mean, mean of squares) of frames without emitters
    rng_exp = np.random.default_rng(0)
    bg_map = param_dict.get('bg_mean_map', None)
    for gidx, chunk in iter_sampled_frames(param_dict['im_br_folder'], n_exp_frames):
        if bg_map is None:
            det = chunk
            fmax = chunk.max(axis=(1, 2))
            sm = smz = smooth3(det)
        else:  # raw movie: remove this file's background (temporal median, local in time) and each frame's level
            det = chunk - np.median(chunk, axis=0)
            det = det - np.median(det.reshape(len(det), -1), axis=1)[:, None, None]
            fmax = np.percentile(det.reshape(len(det), -1), 99, axis=1)  # level statistic for disturbances
            # detect on the noise-normalized image: the noise varies a lot over a raw field (brighter = noisier)
            sd_t = np.maximum(1.4826 * np.median(np.abs(det), axis=0), 0.5)
            sm, smz = smooth3(det), smooth3(det / sd_t)
        if thr is None:
            bg_mu = float(np.median(det))
            sm_mu = float(np.median(smz))
            sm_sd = float(1.4826 * np.median(np.abs(smz - sm_mu)))  # robust: emitters hardly move it
            thr = sm_mu + 12 * sm_sd  # well above the noise peaks (they fall off quickly beyond ~9 sigma)
        frame_idx.append(gidx)
        frame_max.append(fmax)
        quiet = smz.max(axis=(1, 2)) <= thr
        if quiet.any():
            q = det[quiet].astype(np.float64)
            noise_frames.append(np.c_[gidx[quiet], q.mean(axis=(1, 2)), (q ** 2).mean(axis=(1, 2))])
        for i in np.where(~quiet)[0]:
            p = blob_peaks(sm[i], thr, det=smz[i])
            if len(p) > max_peaks_per_frame:  # dense frame: a random subset, so no single frame dominates
                p = rng_exp.choice(p, max_peaks_per_frame, replace=False)
            cand.append((gidx[i], chunk[i], p))
    frame_idx, frame_max = np.concatenate(frame_idx), np.concatenate(frame_max)

    # disturbances (e.g. room light): frames whose max jumps far above that of ordinary frames with
    # emitters; they and their neighbours (onset, afterglow) are left out
    disturbances, ceiling, m0 = find_disturbances(frame_idx, frame_max, disturbance_margin,
                                                  rule='max' if bg_map is None else 'level')
    def disturbed(i):
        return bool(in_ranges([i], disturbances)[0])
    if verbose and disturbances:
        stat = 'frame max' if bg_map is None else 'frame 99th percentile above background'
        print(f'Nsig calibration: {stat} jumps above {ceiling:.0f} (typical {m0:.0f}); left out frames {disturbances}')

    kept_frames = [f for g, f, _ in cand if not disturbed(g)]
    exp_peaks_per_frame = [p for g, _, p in cand if not disturbed(g)]
    exp_peaks = np.concatenate(exp_peaks_per_frame) if exp_peaks_per_frame else np.empty(0)
    if len(exp_peaks) >= 30:  # drop rare very bright events far above the bulk
        exp_peaks = exp_peaks[exp_peaks <= 5 * np.median(exp_peaks)]
    thr_text = f'> {thr:.0f}' if bg_map is None else f'> {thr:.1f} x local noise'
    if len(exp_peaks) < 30:
        print(f'Nsig calibration skipped: only {len(exp_peaks)} emitter peaks ({thr_text}, 3x3-smoothed) '
              f'in {len(frame_idx)} frames outside disturbances.')
        return None, dict(thr=thr, disturbances=disturbances, frame_idx=frame_idx, frame_max=frame_max)

    # background noise of the experiment (frames without emitters, outside disturbances) -> simulation noise
    noise_frames = np.concatenate(noise_frames) if noise_frames else np.empty((0, 3))
    noise_frames = noise_frames[[not disturbed(g) for g in noise_frames[:, 0]]]
    if fit_noise and bg_map is None and len(noise_frames) > 0:
        mu = float(noise_frames[:, 1].mean())
        sd = float(np.sqrt(noise_frames[:, 2].mean() - mu ** 2))
        param_dict['baseline'] = (round(mu - 0.5, 2), round(mu + 0.5, 2))
        param_dict['read_std'] = (round(0.95 * sd, 2), round(1.05 * sd, 2))
        if verbose:
            print(f'Noise from {len(noise_frames)} frames without emitters: mean {mu:.2f}, std {sd:.2f} -> '
                  f'baseline {param_dict["baseline"]}, read_std {param_dict["read_std"]}')

    qs = np.array([0.5, 0.75, 0.9])  # upper quantiles: the lowest peaks sit at the threshold, mixed with the noise tail
    q_exp = np.quantile(exp_peaks, qs)

    model = ImModelTraining(param_dict)
    sampling = Sampling(param_dict)
    rng_state, torch_state = np.random.get_state(), torch.get_rng_state()
    np.random.seed(0)
    torch.manual_seed(0)

    # photon-independent parts of the sample images: A = sum_i E_i, B = sum_i u_i E_i with E_i the
    # blurred unit-photon PSF of emitter i and N_i = lo + (hi - lo) * u_i;  im = shot noise(lo*A + (hi-lo)*B) + readout noise
    A, B, noise, sim_sd = [], [], [], []
    # photons are not used here (unit PSFs); a fixed range keeps the drawn emitters independent of the initial guess
    sampling.Nsig_range = (1000, 2000)
    with torch.no_grad():
        for _ in range(n_sim_images):
            xyzps, _, _ = sampling.xyzp_batch()
            n_em = xyzps.shape[0]
            unit = np.c_[xyzps[:, :3], np.ones(n_em)]
            psfs = model.get_psfs(torch.from_numpy(unit).to(device))
            kernels = model.blur_kernels(n_em).unsqueeze(1)
            # on the CPU: deterministic (cuDNN may pick different algorithms run to run, and the fit is flat enough
            # for last-bit differences to move the result)
            blurred = F.conv2d(psfs.cpu().unsqueeze(0), kernels.cpu(), padding='same', groups=n_em).squeeze(0).numpy()
            u = np.random.rand(n_em)
            A.append(blurred.sum(0))
            B.append(np.tensordot(u, blurred, axes=1))
            if bg_map is not None:  # empirical background, as in ImModelTraining.forward, minus its mean
                H, W = A[-1].shape
                k = np.random.randint(bg_map.shape[0])
                r0 = np.random.randint(0, bg_map.shape[1] - H + 1)
                c0 = np.random.randint(0, bg_map.shape[2] - W + 1)
                mb = bg_map[k, r0:r0 + H, c0:c0 + W] + model.bg_offsets[np.random.randint(len(model.bg_offsets))]
                sb = param_dict['bg_std_map'][k, r0:r0 + H, c0:c0 + W]
                noise.append(np.round(mb + np.random.randn(H, W) * sb) - mb)
                sim_sd.append(np.maximum(sb, 0.5))
                continue
            # readout noise exactly as in ImModelTraining.forward (does not depend on photons)
            bl = model.baseline[0] + np.random.rand() * (model.baseline[1] - model.baseline[0])
            rs = model.read_std[0] + np.random.rand() * (model.read_std[1] - model.read_std[0])
            if model.non_uniform_noise_flag:
                std = model.non_uniform_noise().cpu().numpy() * np.random.rand() * (model.read_std[1] - rs) + rs
            else:
                std = rs
            noise.append(np.round(bl + np.random.randn(*A[-1].shape) * std))
    A, B, noise = np.array(A), np.array(B), np.array(noise)
    sim_sd = np.array(sim_sd) if sim_sd else None
    z_shot = np.random.randn(*A.shape).astype(np.float32)  # fixed draws -> every candidate sees the same noise
    floor = 0 if bg_map is None else -np.inf  # with maps the simulation is compared after removing the background

    def simulate(lo, hi):
        clean = np.clip(lo * A + (hi - lo) * B, 0, None)
        # shot noise: Gaussian approximation with fixed draws (>= 1 photon), so the loss changes smoothly with
        # the photon range (redrawing Poisson numbers would reshuffle all the noise when one pixel changes)
        shot = clean.copy()
        sig = clean > 1
        shot[sig] += np.sqrt(clean[sig]) * z_shot[sig]
        return np.clip(shot + noise, floor, None).astype(np.float32)

    def sim_quantiles(ims, sds=None):
        sm = smooth3(ims)
        smz = sm if sds is None else smooth3(ims / sds)
        peaks = np.concatenate([blob_peaks(sm[k], thr, det=smz[k]) for k in range(len(sm))])
        return (np.quantile(peaks, qs) if len(peaks) >= 10 else None), peaks

    def loss(x):
        lo, hi = np.exp(x)
        if lo >= hi or lo < 100:
            return 1e3
        q, _ = sim_quantiles(simulate(lo, hi), sim_sd)
        if q is None:
            return 1e2
        fit = float(np.mean((np.log(np.maximum(q - sm_mu, 1)) - np.log(q_exp - sm_mu)) ** 2))
        # weak prior hi/lo ~ 3: the dim end is barely seen (below the detection threshold), so without it
        # the lower bound drifts along a flat valley of the fit
        return fit + 0.01 * (np.log(hi / lo) - np.log(3)) ** 2

    # start from a coarse scan of the photon scale (hi = 3 lo), not from the initial guess: the fit has a flat
    # valley, so a start-dependent optimum would change with small changes of the guess
    grid = np.exp(np.linspace(np.log(300), np.log(3e5), 25))
    grid_loss = [loss(np.log([g, 3 * g])) for g in grid]
    g0 = grid[int(np.argmin(grid_loss))]
    best = minimize(loss, np.log([g0, 3 * g0]), method='Nelder-Mead', options=dict(xatol=0.02, fatol=1e-3, maxfev=100))
    history = [dict(grid=grid.tolist(), grid_loss=grid_loss, lo=float(np.exp(best.x[0])), hi=float(np.exp(best.x[1])),
                    err=float(np.sqrt(best.fun)), nfev=int(best.nfev))]

    nsig = (float(np.round(np.exp(best.x[0]) / 100) * 100), float(np.round(np.exp(best.x[1]) / 100) * 100))

    # verification with the full imaging model (random blur, noise, emitters), as the training data will be made
    np.random.seed(1)
    torch.manual_seed(1)
    sampling.Nsig_range = (int(nsig[0]), int(nsig[1]))
    ims, det_ims, det_sd = [], [], []
    with torch.no_grad():
        for _ in range(n_sim_images):
            xyzps, _, _ = sampling.xyzp_batch()
            im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.float32)
            ims.append(im)
            det_ims.append(im if bg_map is None else im - model.last_bg.cpu().numpy())
            if bg_map is not None:
                det_sd.append(np.maximum(model.last_std.cpu().numpy(), 0.5))
    ims = np.array(ims)
    q_sim, sim_peaks = sim_quantiles(np.array(det_ims), np.array(det_sd) if det_sd else None)
    np.random.set_state(rng_state)
    torch.set_rng_state(torch_state)
    err = float(np.max(np.abs(np.log((q_sim - sm_mu) / (q_exp - sm_mu))))) if q_sim is not None else np.inf
    if verbose:
        print(f'Nsig calibration: Nsig_range = ({nsig[0]:.0f}, {nsig[1]:.0f}) from {len(exp_peaks)} experimental peaks '
              f'(3x3-smoothed, {thr_text}); quantiles {qs.tolist()}\n'
              f'  experiment {np.round(q_exp).tolist()}\n'
              f'  simulation {np.round(q_sim).tolist() if q_sim is not None else None} (max log-ratio error {err:.2f})')
    info = dict(bg_mu=bg_mu, thr=thr, err=err, history=history, n_exp_peaks=len(exp_peaks),
                n_sim_peaks=len(sim_peaks), q_exp=q_exp, q_sim=q_sim, disturbances=disturbances,
                baseline=param_dict['baseline'], read_std=param_dict['read_std'])
    if save_dir is not None:
        save_nsig_overlay(save_dir, np.array(kept_frames), exp_peaks, sim_peaks, thr, nsig, err, ims,
                          frame_idx, frame_max, ceiling, disturbances, thr_text)
    return nsig, info


def save_nsig_overlay(save_dir, exp_frames, exp_peaks, sim_peaks, thr, nsig, err, sim_ims,
                      frame_idx, frame_max, ceiling, disturbances, thr_text=None):
    """Save histograms of the simulation over the experiment (jpg) and RGB image overlays (tif).

    exp_frames: experimental frames that contain emitters; sim_ims: simulated images (same intensity space).
    """
    vmax = float(np.quantile(exp_peaks, 0.99))
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
    hi_edge = max(float(np.quantile(exp_frames, 0.9999)), vmax, 100.0)
    lo_edge = min(float(np.quantile(exp_frames, 0.001)), float(np.quantile(sim_ims, 0.001)))
    sample = np.concatenate([exp_frames.ravel()[:100000], sim_ims.ravel()[:100000]])
    if np.all(sample == np.round(sample)):
        # integer pixel values (ADU): bin edges between integers, a whole number of values per bin, otherwise
        # bins narrower than 1 ADU hold no value at all and the histogram shows regular gaps
        width = max(1, int(np.ceil((hi_edge - lo_edge) / 80)))
        bins = np.arange(np.floor(lo_edge) - 0.5, np.ceil(hi_edge) + width, width)
    else:
        bins = np.linspace(lo_edge, hi_edge, 80)
    ax[0].hist(np.clip(exp_frames.ravel(), lo_edge, hi_edge), bins=bins, density=True, alpha=0.55, log=True,
               label='experiment (frames with emitters)')
    ax[0].hist(np.clip(sim_ims.ravel(), lo_edge, hi_edge), bins=bins, density=True, alpha=0.55, log=True, label='simulated')
    ax[0].set_title('pixel values')
    ax[0].legend(fontsize=7)
    pbins = np.linspace(min(0.0, float(exp_peaks.min())), max(float(np.quantile(exp_peaks, 0.99)) * 1.3, thr * 1.5), 60)
    ax[1].hist(np.clip(exp_peaks, 0, pbins[-1]), bins=pbins, density=True, alpha=0.55, label='experiment')
    ax[1].hist(np.clip(sim_peaks, 0, pbins[-1]), bins=pbins, density=True, alpha=0.55, label='simulated')
    ax[1].set_title(f'emitter peaks of the 3x3-smoothed image, above background ({thr_text or f"> {thr:.0f}"})')
    ax[1].legend(fontsize=7)
    ax[2].imshow(np.clip(sim_ims[0], lo_edge, hi_edge), cmap='gray')
    ax[2].set_title(f'a simulated image (Nsig_range = {nsig[0]:.0f}, {nsig[1]:.0f}; quantile error {err:.2f})', fontsize=9)
    ax[2].axis('off')
    ax[3].plot(frame_idx, frame_max, '.', ms=2)
    for a, b in disturbances:
        ax[3].axvspan(a, b, color='r', alpha=0.25)
    if np.isfinite(ceiling):
        ax[3].axhline(ceiling, color='r', ls='--', lw=1)
    ax[3].set_yscale('log')
    ax[3].set_xlabel('frame')
    ax[3].set_title('disturbance statistic per frame (red: left out)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nsig_calibration.jpg'), dpi=120)
    plt.close(fig)

    # overlays: an experimental crop with an emitter (green) under a simulated image (magenta), same intensity scale
    rng = np.random.default_rng(0)
    H, W = sim_ims.shape[1:]
    rgb = []
    for k in range(min(5, len(exp_frames), len(sim_ims))):
        f = exp_frames[rng.integers(len(exp_frames))]
        r, c = np.unravel_index(np.argmax(smooth3(f)), f.shape)
        r0 = int(np.clip(r - H // 2, 0, max(f.shape[0] - H, 0)))
        c0 = int(np.clip(c - W // 2, 0, max(f.shape[1] - W, 0)))
        crop = f[r0:r0 + H, c0:c0 + W]
        crop = np.pad(crop, ((0, H - crop.shape[0]), (0, W - crop.shape[1])))
        g = np.clip((crop - lo_edge) / (hi_edge - lo_edge), 0, 1)
        m = np.clip((sim_ims[k] - lo_edge) / (hi_edge - lo_edge), 0, 1)
        rgb.append((np.stack([m, g, m], axis=-1) * 255).astype(np.uint8))
    if rgb:
        tifffile.imwrite(os.path.join(save_dir, 'nsig_overlay.tif'), np.stack(rgb), photometric='rgb')


def mu_std_p(param_dict, noise_dict):

    im_br_folder = param_dict['im_br_folder']
    num = noise_dict['num_ims']
    # noise_roi = noise_dict['noise_roi']
    snr_roi = noise_dict['snr_roi']
    max_pv = noise_dict['max_pv']

    # last `num` frames: at the end of the video, probably with sparse molecules. It's ok if num > total frames
    ims = read_last_frames(im_br_folder, num)

    ims = ims[:, snr_roi[0]:snr_roi[2], snr_roi[1]:snr_roi[3]]

    max_map = np.max(ims, axis=0)
    mean_map = np.mean(ims, axis=0)

    r_idx, c_idx = np.unravel_index(np.argmin(mean_map), mean_map.shape)
    bg_pixel = ims[:, r_idx, c_idx]
    mu, std = np.mean(bg_pixel), np.std(bg_pixel)

    if max_pv == 0:
        exp_maxv = np.max(max_map)  # if max_pv is 0 in the GUI
    else:
        exp_maxv = max_pv  # detect max_pv in the selected ROI

    print(f'Detected MPV: {exp_maxv}.')

    model = ImModelBase(param_dict)

    photon_count = 1e4
    xyzps = np.array([[0, 0, (param_dict['zrange'][0]+param_dict['zrange'][1])/2, photon_count]])  # take the middle z
    xyzps = torch.from_numpy(xyzps).to(param_dict['device'])
    ims = model.get_psfs(xyzps).cpu().numpy()

    maxvs = np.max(ims, axis=(1, 2))
    mv = np.mean(maxvs) + mu  # model.get_psfs doesn't include noise yet
    p = photon_count/mv * exp_maxv

    return mu, std, p, exp_maxv


def training_data_func(param_dict):

    device = param_dict['device']
    # imaging model
    model = ImModelTraining(param_dict)
    # sampling model
    sampling = Sampling(param_dict)
    sampling.show_volume()  # plot volume

    # start
    td_folder = param_dict['td_folder']
    if os.path.exists(td_folder):  # delete the directory if it exists
        shutil.rmtree(td_folder)
    x_folder = os.path.join(td_folder, 'x')
    os.makedirs(x_folder)  # make the folder for training data

    # labels_dict for training
    labels_dict = {}
    labels_dict['volume_size'] = (param_dict['D'], param_dict['HH'], param_dict['WW'])
    labels_dict['us_factor'] = param_dict['us_factor']
    labels_dict['blob_r'] = sampling.blob_r  # radius of each 3D blob representing an emitter in space
    labels_dict['blob_maxv'] = sampling.blob_maxv  # maximum value of blobs

    ntrain = param_dict['n_ims']
    for i in range(ntrain):
        xyzps, xyz_ids, blob3d = sampling.xyzp_batch()
        im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.uint16)
        if param_dict['project_01']:
            im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)

        x_name = str(i).zfill(5) + '.tif'
        io.imsave(os.path.join(x_folder, x_name), im, check_contrast=False)
        labels_dict[x_name] = {'xyz_ids': xyz_ids, 'blob3d': blob3d, 'xyzps': xyzps}

        if i % (ntrain // 10) == 0:
            print('Training image [%d / %d]' % (i + 1, ntrain))
    print('Training image [%d / %d]' % (ntrain, ntrain))

    y_file = os.path.join(td_folder, r'y.pickle')
    with open(y_file, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    param_file = os.path.join(td_folder, r'param.pickle')
    with open(param_file, 'wb') as handle:
        pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def training_func(param_dict, training_dict):
    np.random.seed(66)
    torch.manual_seed(88)

    device = param_dict['device']
    torch.backends.cudnn.benchmark = True

    td_folder = param_dict['td_folder']
    path_save = param_dict['path_save']
    os.makedirs(path_save, exist_ok=True)

    batch_size = training_dict['batch_size']
    lr = training_dict['lr']
    num_epochs = training_dict['num_epochs']

    params_train = {'batch_size': batch_size, 'shuffle': True}
    params_validate = {'batch_size': batch_size, 'shuffle': True}

    x_folder = os.path.join(td_folder, 'x')
    x_list = os.listdir(x_folder)
    num_x = len(x_list)
    with open(os.path.join(td_folder, 'y.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    partition = {'train': x_list[:int(num_x * 0.9)], 'validate': x_list[int(num_x * 0.9):]}
    train_ds = MyDataset(x_folder, partition['train'], labels)
    train_dl = DataLoader(train_ds, **params_train)
    validate_ds = MyDataset(x_folder, partition['validate'], labels)
    validate_dl = DataLoader(validate_ds, **params_validate)

    D, us_factor, maxv = labels['volume_size'][0], labels['us_factor'], labels['blob_maxv']
    model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# of trainable parameters: {n_params}')

    optimizer = Adam(list(model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1,
                                  min_lr=1e-6)
    if param_dict['us_factor'] == 1:
        # my_loss_func = torch.nn.MSELoss()
        my_loss_func = KDE_loss3D(sigma=0.5, device=device)
    else:
        my_loss_func = KDE_loss3D(sigma=0.5*(param_dict['us_factor']/2), device=device)  # 0.5-2, 1.0-4

    trainer = TorchTrainer(model, my_loss_func, optimizer, lr_scheduler=scheduler, device=device)

    time_now = datetime.today().strftime('%m-%d_%H-%M')
    net_file = 'net_' + time_now + '.pt'
    checkpoints = dict(file_name=os.path.join(path_save, net_file),
                       net=Net(D=D, us_factor=us_factor, maxv=maxv),
                       state_dict=None,
                       note=' '
                       )

    t0 = time.time()
    fit_results = trainer.fit(train_dl, validate_dl, num_epochs=num_epochs, checkpoints=checkpoints, early_stopping=4)
    fit_file = 'fit_' + time_now + '.pickle'
    with open(os.path.join(path_save, fit_file), 'wb') as handle:
        pickle.dump(fit_results, handle)

    t1 = time.time()
    print(f'training results in {net_file} and {fit_file}')
    print(f'finished training in {t1 - t0}s.')

    return net_file, fit_file


def inference_func1(param_dict, test_idx, fig_flag=True):  # simulation and try one exp image

    np.random.seed(11)
    torch.manual_seed(11)

    device = param_dict['device']
    path_save = param_dict['path_save']
    net_file = param_dict['net_file']
    fit_file = param_dict['fit_file']

    # training performance
    with open(os.path.join(path_save, fit_file), 'rb') as handle:
        fit_result = pickle.load(handle)
    train_loss = fit_result.train_loss
    test_loss = fit_result.test_loss

    if fig_flag:
        num_epochs = len(train_loss)
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(num_epochs), train_loss)
        plt.plot(np.arange(num_epochs), test_loss)
        plt.title('training loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        loss_file = out_path(param_dict, 'loss_curves.jpg')
        plt.savefig(loss_file, bbox_inches='tight', dpi=300)
        plt.clf()

        print(f'Training loss curves: {loss_file}')

    checkpoint = torch.load(os.path.join(path_save, net_file), map_location=device)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])

    # simulated data
    model = ImModelTraining(param_dict)
    sampling = Sampling(param_dict)
    volume2xyz = Volume2XYZ(param_dict)

    # simulation
    xyzps, _, _ = sampling.xyzp_batch()
    im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.float32)
    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min()))

    with torch.no_grad():
        net.eval()
        vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
    xyz_rec, conf_rec = volume2xyz(vol)

    if xyz_rec is not None:
        xyz_gt = xyzps[:, :-1]
        jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 0.1)
        jaccard_index, RMSE_xy, RMSE_z = np.round(jaccard_index, decimals=2), np.round(RMSE_xy*1000, decimals=2), np.round(RMSE_z*1000, decimals=2)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], c='b', marker='o', label='GT', depthshade=False)
        ax.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], c='r', marker='^', label='Rec', depthshade=False)
        ax.set_xlabel('X [um]')
        ax.set_ylabel('Y [um]')
        ax.set_zlabel('Z [um]')
        if RMSE_xy is not None:
            plt.title(f'Found {xyz_rec.shape[0]} / {xyz_gt.shape[0]}, j_idx: {jaccard_index}, r_xy: {RMSE_xy} nm, r_z: {RMSE_z} nm')
        else:
            plt.title(f'Found {xyz_rec.shape[0]} emitters out of {xyz_gt.shape[0]}')
        plt.legend()
        plt.savefig(out_path(param_dict, 'sim_loc_gt_rec.jpg'), dpi=300)
        plt.clf()

        nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
        psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()
        im_rec = np.sum(psfs_rec, axis=0)
        im_rec = (im_rec-im_rec.min())/(im_rec.max()-im_rec.min())
        im = (im-im.min())/(im.max()-im.min())

        fig = plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(im, cmap='gray')
        plt.title('im')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(im_rec, cmap='gray')
        plt.title('im_rec')
        plt.axis('off')

        mask = np.max(psfs_rec, axis=0)
        mask = (mask-mask.min())/(mask.max()-mask.min())
        mask = 1-mask
        transparency = 0.2+mask*0.8
        im_overlay = np.stack((im, im, im, transparency), axis=-1)
        im_overlay[:, :, 1] = im_overlay[:, :, 1] * mask
        plt.subplot(1, 3, 3)
        plt.imshow(im_overlay)
        plt.title('overlay')
        plt.axis('off')

        sim_file = out_path(param_dict, 'sim_im_gt_rec.jpg')
        plt.savefig(sim_file, bbox_inches='tight', dpi=300)
        plt.clf()

        print(f'Network inference on simulated an image: {sim_file}')

    exp_imgs_path = param_dict['im_br_folder']

    im = read_frame(exp_imgs_path, test_idx).astype(np.float32)  # read the test frame (global index)
    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
    with torch.no_grad():
        net.eval()
        vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
    xyz_rec, conf_rec = volume2xyz(vol)


    H, W = im.shape
    param_dict['H'], param_dict['W'] = H, W
    model = ImModelTraining(param_dict)

    if H > param_dict['phase_mask'].shape[0] or W > param_dict['phase_mask'].shape[1]:
        sf = max(H // param_dict['phase_mask'].shape[0]+1, W // param_dict['phase_mask'].shape[1]+1)
        param_dict['ps_BFP'] /= sf
        phase_mask = param_dict['phase_mask']

        HW = np.floor(param_dict['f_4f'] * param_dict['lamda'] / (param_dict['ps_camera'] * param_dict['ps_BFP']))  # simulation size
        HW = int(HW + 1 - (HW % 2))  # make it odd

        phase_mask = interpolate(torch.tensor(phase_mask).unsqueeze(0).unsqueeze(1), size=(HW, HW))
        param_dict['phase_mask'] = phase_mask[0, 0].numpy()
        model = ImModelTraining(param_dict)

    nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
    psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()

    im_rec = np.sum(psfs_rec, axis=0)
    im_rec = (im_rec - im_rec.min()) / (im_rec.max() - im_rec.min())

    im = (im - im.min()) / (im.max() - im.min())

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap='gray')
    plt.title('im')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_rec, cmap='gray')
    plt.title(f'im_rec, found {xyz_rec.shape[0]} emitters')
    plt.axis('off')

    mask = np.max(psfs_rec, axis=0)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = 1 - mask
    transparency = 0.2 + mask * 0.8

    im_overlay = np.stack((im, im, im, transparency), axis=-1)

    # mask = mask<0.1
    im_overlay[:, :, 1] = im_overlay[:, :, 1] * mask
    plt.subplot(1, 3, 3)
    plt.imshow(im_overlay)
    plt.title('overlay')
    plt.axis('off')


    exp_file = out_path(param_dict, 'exp_im_gt_rec.jpg')
    plt.savefig(exp_file, bbox_inches='tight', dpi=300)

    print(f'Network inference on a test experimental image: {exp_file}')


def inference_func2(param_dict):

    device = param_dict['device']
    path_save = param_dict['path_save']
    net_file = param_dict['net_file']
    checkpoint = torch.load(os.path.join(path_save, net_file), map_location=device)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])

    # simulated data
    volume2xyz = Volume2XYZ(param_dict)

    exp_imgs_path = param_dict['im_br_folder']
    num_imgs = sum(n for _, n in frame_counts(exp_imgs_path))

    tall_start = time.time()
    results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
    with torch.no_grad():
        net.eval()
        for im_ind, im_raw in iter_frames(exp_imgs_path):
            # print current image number
            print('Processing Image [%d/%d]' % (im_ind + 1, num_imgs))

            tfrm_start = time.time()

            im = im_raw.astype(np.float32)
            if param_dict['project_01']:
                im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
            vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
            xyz_rec, conf_rec = volume2xyz(vol)

            tfrm_end = time.time() - tfrm_start

            # if this is the first image, get the dimensions and the relevant center for plotting
            H, W = im.shape
            ch, cw = H / 2, W / 2

            # if prediction is empty then set number fo found emitters to 0
            # otherwise generate the frame column and append results for saving
            if xyz_rec is None:
                nemitters = 0
            else:
                nemitters = xyz_rec.shape[0]
                frm_rec = (im_ind + 1) * np.ones(nemitters)
                xnm = (xyz_rec[:, 0] + cw * param_dict['vs_xy']*param_dict['us_factor']) * 1000
                ynm = (xyz_rec[:, 1] + ch * param_dict['vs_xy']*param_dict['us_factor']) * 1000
                znm = (xyz_rec[:, 2]) * 1000  # make sure they are above 0
                xyz_save = np.c_[xnm, ynm, znm]

                results = np.vstack((results, np.column_stack((frm_rec, xyz_save, conf_rec))))

            print('Single frame complete in {:.6f}s, found {:d} emitters'.format(tfrm_end, nemitters))

    # print the time it took for the entire analysis
    tall_end = time.time() - tall_start
    print('=' * 50)
    print('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        tall_end // 3600, np.floor((tall_end / 3600 - tall_end // 3600) * 60), tall_end % 60))
    print('=' * 50)

    # write the results to a csv file named "localizations.csv" under the output folder
    row_list = results.tolist()

    time_now = datetime.today().strftime('%m-%d_%H-%M')
    file_name = out_path(param_dict, 'localizations_' + time_now + '.csv')
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)

    print(f'Localization list: {file_name}')

    return file_name


def match_localizations_hungarian(locs_a, locs_b, max_distance=None):
    """Match two sets of localizations with the Hungarian algorithm and return mean distance.

    Args:
        locs_a: (N, D) array of localizations in nm.
        locs_b: (M, D) array of localizations in nm.
        max_distance: optional distance cutoff (nm); matched pairs beyond this are excluded.

    Returns:
        mean_dist: mean Euclidean distance of matched pairs in nm.
        dists: (K,) array of individual matched distances after filtering.
    """
    C = cdist(locs_a, locs_b, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(C)
    dists = C[row_ind, col_ind]
    if max_distance is not None:
        dists = dists[dists <= max_distance]
    return float(np.mean(dists)), dists



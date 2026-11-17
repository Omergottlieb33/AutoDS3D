import argparse
import pickle
import os
import torch
import re
import tifffile
import numpy as np
import pandas as pd
from DS3Dplus.ds3d_utils import ImModelBase, ImModelTraining, Sampling, calc_jaccard_rmse, Volume2XYZ
from DS3Dplus.ds3d_utils import LON as Net
from tqdm import tqdm
import csv

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for trained DS3D+ model.")
    parser.add_argument('--checkpoints_path', type=str, required=True,
                        help='Path to the trained model checkpoints.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the inference on (e.g., "cpu" or "cuda").')
    parser.add_argument('--images_path', type=str, required=True,
                        help='Folder of multi-frame .tif stacks (frame range in the name, e.g. *_00000-02878.tif).')
    parser.add_argument('--params_path', type=str, required=True,
                        help='Path to the parameters file used during training.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the inference results (CSV file).')
    parser.add_argument('--threshold', type=float, default=40,
                        help='Detection threshold on the predicted volume, in units of '
                             'blob_maxv (1000). Sweep this rather than trusting the default.')
    parser.add_argument('--border', type=int, default=0,
                        help='Discard localizations within this many pixels of the image '
                             'edge. The network never sees emitters inside the psf_half_size '
                             'buffer (20 px) during training, so that band is extrapolation.')
    parser.add_argument('--max_tifs', type=int, default=None,
                        help='Process only the first N tif files (for quick tests).')
    return parser.parse_args()

def load_model(checkpoints_path: str, device: torch.device):
    # load trained model
    # Allowlist the model class for secure unpickling on PyTorch >= 2.6
    try:
        torch.serialization.add_safe_globals([Net])
        checkpoint = torch.load(checkpoints_path, map_location=device)
    except Exception:
        # Fallback for environments without safe globals or if loading still fails
        checkpoint = torch.load(
            checkpoints_path, map_location=device, weights_only=False)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()
    return net

def inference(checkpoints_path: str, device: torch.device, images_path: str, params_path: str, save_path: str, save_frame_csv: bool = False, threshold: float = 40, border: int = 0, max_tifs: int = None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # load model
    model = load_model(checkpoints_path, device)
    # load parameters
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    params['device'] = device
    params['threshold'] = threshold
    volume2xyz = Volume2XYZ(params)
    ps_xy = params['vs_xy']*params['us_factor']
    #ps_xy = params['ps_xy'] # FOV size, camera pixel size/magnification
    tif_files = sorted(f for f in os.listdir(images_path)
                       if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.'))
    if max_tifs is not None:
        tif_files = tif_files[:max_tifs]
    num_tifs = len(tif_files)
    next_frame = 1  # fallback numbering (1-based) when a name has no frame range
    with open(save_path, 'w', newline='') as result_file, torch.no_grad():
        writer = csv.writer(result_file)
        writer.writerow(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
        for tif_idx, tif_name in enumerate(tif_files):
            stack = tifffile.imread(os.path.join(images_path, tif_name))
            if stack.ndim == 2:
                stack = stack[np.newaxis]
            # the first index in the name ("..._frames_02879-05757.tif") is the 0-based global
            # index of the stack's first frame; it keeps frame numbers consistent across files
            m = re.search(r'(\d+)-(\d+)(?=\.tiff?$)', tif_name, re.IGNORECASE)
            first_frame = int(m.group(1)) + 1 if m else next_frame
            next_frame = first_frame + len(stack)
            H, W = stack.shape[1:]
            ch, cw = H / 2, W / 2
            for i in tqdm(range(len(stack)), desc=f'{tif_name} [{tif_idx + 1}/{num_tifs}]'):
                im = stack[i].astype(np.float32)
                if params['project_01']:
                    im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
                vol = model(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
                xyz_rec, conf_rec = volume2xyz(vol)
                # no prediction -> nothing to write for this frame
                if xyz_rec is None:
                    continue
                xnm = (xyz_rec[:, 0] + cw * ps_xy) * 1000
                ynm = (xyz_rec[:, 1] + ch * ps_xy) * 1000
                znm = (xyz_rec[:, 2]) * 1000

                if border > 0:
                    # keep only detections at least `border` px from the edge
                    keep = ((xnm / (ps_xy * 1000) >= border) & (xnm / (ps_xy * 1000) <= W - border) &
                            (ynm / (ps_xy * 1000) >= border) & (ynm / (ps_xy * 1000) <= H - border))
                    xnm, ynm, znm, conf_rec = xnm[keep], ynm[keep], znm[keep], conf_rec[keep]

                frame_no = first_frame + i
                frm_rec = frame_no * np.ones(xnm.shape[0])
                writer.writerows(np.column_stack((frm_rec, xnm, ynm, znm, conf_rec)).tolist())

                if save_frame_csv:
                    frame_csv_path = os.path.join(os.path.dirname(save_path), f'{frame_no}.csv')
                    with open(frame_csv_path, 'w', newline='') as frame_file:
                        frame_writer = csv.writer(frame_file)
                        frame_writer.writerow(['x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
                        frame_writer.writerows(np.column_stack((xnm, ynm, znm, conf_rec)).tolist())
    print(f'{save_path} is saved.')

if  __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    inference(args.checkpoints_path, device, args.images_path, args.params_path, args.save_path,
              threshold=args.threshold, border=args.border, max_tifs=args.max_tifs)
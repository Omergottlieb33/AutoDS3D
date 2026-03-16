import argparse
import pickle
import os
import torch
from skimage.io import imread
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
                        help='Path to the folder containing input images for inference.')
    parser.add_argument('--params_path', type=str, required=True,
                        help='Path to the parameters file used during training.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the inference results (CSV file).')
    parser.add_argument('--patch_size', type=int, default=120,
                        help='Size of the sliding window patch (e.g. 128).')
    parser.add_argument('--overlap', type=int, default=16,
                        help='Overlap size between patches (e.g. 16).')
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

def inference(checkpoints_path: str, device: torch.device, images_path: str, params_path: str, save_path: str, save_frame_csv: bool = False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # load model
    model = load_model(checkpoints_path, device)
    # load parameters
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    params['device'] = device
    params['threshold'] = 40  # set a fixed threshold for inference
    volume2xyz = Volume2XYZ(params)
    ps_xy = params['vs_xy']*params['us_factor']
    #ps_xy = params['ps_xy'] # FOV size, camera pixel size/magnification
    img_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f)) and not f.startswith('.')]
    try:
        sorted_img_names = sorted(img_files, key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        print("Filenames are not integers. Sorting alphabetically.")
        sorted_img_names = sorted(img_files)
    num_imgs = len(sorted_img_names)
    results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
    with torch.no_grad():
        for idx, img_name in tqdm(enumerate(sorted_img_names)):
            im = imread(os.path.join(images_path, img_name)).astype(np.float32)
            if params['project_01']:
                im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
            vol = model(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
            xyz_rec, conf_rec = volume2xyz(vol)
            # if this is the first image, get the dimensions and the relevant center for plotting
            if idx == 0:
                H, W = im.shape
                ch, cw = H / 2, W / 2
            # if prediction is empty then set number fo found emitters to 0
            # otherwise generate the frame column and append results for saving
            if xyz_rec is None:
                nemitters = 0
            else:
                nemitters = xyz_rec.shape[0]
                frm_rec = (idx + 1) * np.ones(nemitters)
                
                xnm = (xyz_rec[:, 0] + cw * ps_xy) * 1000
                ynm = (xyz_rec[:, 1] + ch * ps_xy) * 1000
                znm = (xyz_rec[:, 2]) * 1000
                xyz_save = np.c_[xnm, ynm, znm]
                
                results = np.vstack((results, np.column_stack((frm_rec, xyz_save, conf_rec))))
                log_interval = max(1, num_imgs // 10)
                if idx % log_interval == 0:
                    print('Processed Image [%d/%d]' % (idx + 1, num_imgs))
                    # print status
                    print('Single frame complete in found {:d} emitters'.format(nemitters))
                if save_frame_csv:
                    frame_csv_path = os.path.join(os.path.dirname(save_path), f'{img_name.split(".")[0]}.csv')
                    with open(frame_csv_path, 'w', newline='') as frame_file:
                        frame_writer = csv.writer(frame_file)
                        frame_writer.writerow(['x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
                        frame_writer.writerows(np.column_stack((xnm, ynm, znm, conf_rec)).tolist())
                    print(f'{frame_csv_path} is saved.')
    # save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results.tolist())
    print(f'{save_path} is saved.')

if  __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    inference(args.checkpoints_path, device, args.images_path, args.params_path, args.save_path, True)
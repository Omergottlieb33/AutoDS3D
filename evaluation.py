import argparse
import pickle
import os
import torch
from skimage.io import imread
import numpy as np
import pandas as pd
from DS3Dplus.ds3d_utils import ImModelBase, ImModelTraining, Sampling, calc_jaccard_rmse, Volume2XYZ
from DS3Dplus.ds3d_utils import LON as Net


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate DS3D+ model")
    parser.add_argument('--training_results_path', type=str,
                        required=True, help='Path to the evaluation data')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data')
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--blob_r', type=float, default=2.0,
                        help='Blob radius for evaluation')
    parser.add_argument('--threshold', type=float,
                        default=40, help='Threshold for evaluation')
    parser.add_argument('--jaccard_threshold', type=float,
                        default=0.1, help='Jaccard threshold for evaluation')
    return parser.parse_args()


def get_image_and_aberrated_tensors(images_path: str, img_name: str, param_dict: dict):
    img_path = os.path.join(images_path, img_name)
    im = imread(img_path)
    im = im[np.newaxis, :, :].astype(np.float32)
    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
    im_tensor = torch.from_numpy(im).unsqueeze(0).to(param_dict['device'])
    # aberrated
    name, ext = os.path.splitext(img_name)
    abr_img_name = f"{name}_abr{ext}"
    img_path_abr = os.path.join(os.path.dirname(
        images_path), f"abr/{abr_img_name}")
    im_abr = imread(img_path_abr)
    im_abr = im_abr[np.newaxis, :, :].astype(np.float32)
    if param_dict['project_01']:
        im_abr = ((im_abr - im_abr.min()) /
                  (im_abr.max() - im_abr.min())).astype(np.float32)
    im_tensor_abr = torch.from_numpy(
        im_abr).unsqueeze(0).to(param_dict['device'])
    return im_tensor, im_tensor_abr


def evaluate_model_on_aberration_pairs(training_results_path, test_data_dir, device, blob_r, threshold, jaccard_threshold):
    # load trained model
    def _find_latest_pt_file(directory: str) -> str:
        pt_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith('.pt')
        ]
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {directory}")
        # Prefer files that start with 'net_' if present; otherwise pick latest by mtime
        net_pt_files = [p for p in pt_files if os.path.basename(p).startswith('net_')]
        candidates = net_pt_files if net_pt_files else pt_files
        return max(candidates, key=os.path.getmtime)

    checkpoint_path = _find_latest_pt_file(training_results_path)
    # Allowlist the model class for secure unpickling on PyTorch >= 2.6
    try:
        torch.serialization.add_safe_globals([Net])
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        # Fallback for environments without safe globals or if loading still fails
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    # load test data
    with open(os.path.join(test_data_dir, 'y.pickle'), 'rb') as handle:
        localizations = pickle.load(handle)
    x_folder = os.path.join(test_data_dir, 'x/orig')
    with open(os.path.join(test_data_dir, 'param.pickle'), 'rb') as handle:
        param_dict_test_data = pickle.load(handle)
    sorted_img_names = sorted(os.listdir(
        x_folder), key=lambda x: int(os.path.splitext(x)[0]))
    param_dict_test_data['device'] = device
    param_dict_test_data['blob_r'] = blob_r
    param_dict_test_data['threshold'] = threshold
    volume2xyz = Volume2XYZ(param_dict_test_data)
    # evaluate
    results_df_clean = pd.DataFrame({
        'Image': pd.Series(dtype='string'),
        'Jaccard Index': pd.Series(dtype='float64'),
        'RMSE_xy (nm)': pd.Series(dtype='float64'),
        'RMSE_z (nm)': pd.Series(dtype='float64'),
    })
    results_df_aberrated = pd.DataFrame({
        'Image': pd.Series(dtype='string'),
        'Jaccard Index': pd.Series(dtype='float64'),
        'RMSE_xy (nm)': pd.Series(dtype='float64'),
        'RMSE_z (nm)': pd.Series(dtype='float64'),
    })
    for img_name in sorted_img_names:
        xyzps_gt = localizations[img_name]
        xyz_gt = xyzps_gt['xyzps'][:, :-1]
        if xyzps_gt['abr_rmse'] is None:
            aberration_flag = False
        else:
            aberration_flag = True
        # tensors
        im_tensor, im_tensor_abr = get_image_and_aberrated_tensors(
            x_folder, img_name, param_dict_test_data)
        with torch.no_grad():
            volume_pred_clean = net(im_tensor).to(device)
        xyz_pred_clean, _ = volume2xyz(volume_pred_clean)
        jaccard_clean, rmse_xy_clean, rmse_z_clean, _ = calc_jaccard_rmse(
            xyz_gt, xyz_pred_clean, jaccard_threshold)
        results_df_clean = pd.concat([results_df_clean, pd.DataFrame({
            'Image': [img_name],
            'Jaccard Index': [jaccard_clean],
            'RMSE_xy (nm)': [rmse_xy_clean],
            'RMSE_z (nm)': [rmse_z_clean],
        })], ignore_index=True)
        if aberration_flag:
            with torch.no_grad():
                volume_pred_abr = net(im_tensor_abr)
            xyz_pred_abr, _ = volume2xyz(volume_pred_abr)
            jaccard_abr, rmse_xy_abr, rmse_z_abr, _ = calc_jaccard_rmse(
                xyz_gt, xyz_pred_abr, jaccard_threshold)
            results_df_aberrated = pd.concat([results_df_aberrated, pd.DataFrame({
                'Image': [img_name],
                'Jaccard Index': [jaccard_abr],
                'RMSE_xy (nm)': [rmse_xy_abr],
                'RMSE_z (nm)': [rmse_z_abr],
            })], ignore_index=True)
    # save results
    results_df_clean.to_csv(os.path.join(
        training_results_path, 'results_clean.csv'), index=False)
    results_df_aberrated.to_csv(os.path.join(
        training_results_path, 'results_aberrated.csv'), index=False)


if __name__ == "__main__":
    args = get_args()
    evaluate_model_on_aberration_pairs(args.training_results_path, args.test_data_dir,
                                       args.device, args.blob_r, args.threshold, args.jaccard_threshold)

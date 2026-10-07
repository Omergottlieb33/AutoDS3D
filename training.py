import contextlib
import ctypes
import io
import pickle
import torch
import numpy as np
from skimage import io as skio
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from DS3Dplus.training_utils import TorchTrainer
from datetime import datetime
from DS3Dplus.ds3d_utils import MyDataset, KDE_loss3D, Volume2XYZ, calc_jaccard_rmse
from DS3Dplus.ds3d_utils import LON as Net
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
import os
import time
import argparse

np.random.seed(66)
torch.manual_seed(88)


def disable_transparent_hugepages():
    """Opt this process (and its forked DataLoader workers) out of THP.

    The host's physical memory is too fragmented to assemble 2 MB huge pages, so
    every large fresh allocation in the input pipeline enters direct compaction,
    fails, and stalls. That made epochs ~8x slower with the GPU mostly idle.
    """
    PR_SET_THP_DISABLE = 41
    try:
        ctypes.CDLL('libc.so.6').prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)
    except OSError:
        pass  # not Linux, or no libc: nothing to opt out of


#disable_transparent_hugepages()


def get_args():
    parser = argparse.ArgumentParser(description="Train DS3D+ model")
    parser.add_argument('--x_folder_path', type=str, required=True,
                        help='Path to the folder containing input images')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the training data')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training (cuda or cpu)')
    # training recipe; the defaults reproduce the original runs
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (decoupled for adamw)')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'],
                        help='plateau: ReduceLROnPlateau(factor 0.1, patience 1); cosine: linear warmup + cosine decay')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Linear warmup epochs (cosine scheduler only)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Final / minimum learning rate')
    parser.add_argument('--early_stopping', type=int, default=4,
                        help='Stop after this many epochs without improvement; 0 disables')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Max gradient norm; 0 disables clipping')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA of weights for validation and checkpoints (e.g. 0.999); 0 disables')
    parser.add_argument('--val_jaccard_images', type=int, default=0,
                        help='Validation images decoded each epoch to log Jaccard; 0 disables')
    parser.add_argument('--val_threshold', type=float, default=40,
                        help='Decode threshold for the validation Jaccard')
    parser.add_argument('--select_by', type=str, default='loss', choices=['loss', 'jaccard'],
                        help='Checkpoint / early-stopping criterion (jaccard needs --val_jaccard_images)')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    if args.select_by == 'jaccard' and args.val_jaccard_images <= 0:
        parser.error('--select_by jaccard needs --val_jaccard_images > 0')
    return args


def make_val_jaccard_fn(x_folder, image_ids, labels, param_dict, device, threshold, blob_r, radius=0.1):
    """Mean Jaccard (matching radius in microns) of a model on a fixed set of validation images."""
    decode = Volume2XYZ(dict(param_dict, device=device, threshold=threshold, blob_r=blob_r))

    def val_jaccard(model):
        was_training = model.training
        model.eval()
        scores = []
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):  # silence "Empty Prediction!"
            for image_id in image_ids:
                x = skio.imread(os.path.join(x_folder, image_id)).astype(np.float32)
                xyz_pred, _ = decode(model(torch.from_numpy(x)[None, None].to(device)))
                jaccard, _, _, _ = calc_jaccard_rmse(labels[image_id]['xyzps'][:, :3], xyz_pred, radius)
                scores.append(jaccard)
        model.train(was_training)
        return float(np.mean(scores))
    return val_jaccard


def train_model(x_folder_path, data_dir, save_path, epochs, batch_size, lr, device, optimizer='adam',
                weight_decay=0.0, scheduler='plateau', warmup_epochs=2, min_lr=1e-6, early_stopping=4,
                grad_clip=0.0, ema_decay=0.0, val_jaccard_images=0, val_threshold=40, select_by='loss',
                num_workers=8):
    train_args = {k: v for k, v in locals().items()}  # saved with the checkpoint
    os.makedirs(save_path, exist_ok=True)
    loader_kwargs = {'num_workers': num_workers,
                     'pin_memory': True, 'persistent_workers': num_workers > 0}
    params_train = {'batch_size': batch_size, 'shuffle': True, **loader_kwargs}
    params_validate = {'batch_size': batch_size,
                       'shuffle': True, **loader_kwargs}

    x_folder = x_folder_path
    # Sort then shuffle with a fixed seed: os.listdir order is arbitrary, and the
    # generator writes the un-aberrated samples as a contiguous block at the end
    # (aberration_split), so an unshuffled split can put them all in validation.
    x_list = sorted(os.listdir(x_folder))
    np.random.RandomState(66).shuffle(x_list)
    num_x = len(x_list)
    with open(os.path.join(data_dir, 'y.pickle'), 'rb') as handle:
        labels = pickle.load(handle)
    with open(os.path.join(data_dir, 'param.pickle'), 'rb') as handle:
        param_dict = pickle.load(handle)

    partition = {'train': x_list[:int(
        num_x*0.9)], 'validate': x_list[int(num_x*0.9):]}
    train_ds = MyDataset(x_folder, partition['train'], labels)
    train_dl = DataLoader(train_ds, **params_train)
    validate_ds = MyDataset(x_folder, partition['validate'], labels)
    validate_dl = DataLoader(validate_ds, **params_validate)

    D, us_factor, maxv = labels['volume_size'][0], labels['us_factor'], labels['blob_maxv']
    model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# of trainable parameters: {n_params}')

    if optimizer == 'adamw':
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == 'cosine':
        cosine = CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1), eta_min=min_lr)
        if warmup_epochs > 0:
            warmup = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
            lr_scheduler = SequentialLR(opt, [warmup, cosine], milestones=[warmup_epochs])
        else:
            lr_scheduler = cosine
    else:
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=1, min_lr=min_lr)
    if param_dict['us_factor'] == 1:
        my_loss_func = torch.nn.MSELoss()
    else:
        my_loss_func = KDE_loss3D(
            # 0.5-2, 1.0-4
            sigma=0.5*(param_dict['us_factor']/2), device=device)

    trainer = TorchTrainer(
        model,
        my_loss_func,
        opt,
        lr_scheduler=lr_scheduler,
        device=device,
        grad_clip=grad_clip,
        ema_decay=ema_decay
    )

    val_metric_fn = None
    if val_jaccard_images > 0:
        val_metric_fn = make_val_jaccard_fn(x_folder, partition['validate'][:val_jaccard_images], labels,
                                            param_dict, device, val_threshold, labels['blob_r'])

    time_now = datetime.today().strftime('%m-%d_%H-%M')
    net_file = 'net_'+time_now+'.pt'
    checkpoints = dict(file_name=os.path.join(save_path, net_file),
                       net=Net(D=D, us_factor=us_factor, maxv=maxv),
                       state_dict=None,
                       note=' ',
                       train_args=train_args
                       )

    t0 = time.time()
    fit_results = trainer.fit(
        train_dl, validate_dl, num_epochs=epochs, checkpoints=checkpoints, early_stopping=early_stopping,
        val_metric_fn=val_metric_fn, select_by_val_metric=(select_by == 'jaccard'))

    fit_file = 'fit_'+time_now+'.pickle'
    with open(os.path.join(save_path, fit_file), 'wb') as handle:
        pickle.dump(fit_results, handle)

    t1 = time.time()

    print(f'finished training in {t1-t0}s.')


if __name__ == "__main__":
    args = get_args()
    train_model(**vars(args))

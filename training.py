import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from DS3Dplus.training_utils import TorchTrainer
from datetime import datetime
from DS3Dplus.ds3d_utils import MyDataset, KDE_loss3D
from DS3Dplus.ds3d_utils import LON as Net
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import argparse

np.random.seed(66)
torch.manual_seed(88)

def get_args():
    parser = argparse.ArgumentParser(description="Train DS3D+ model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (cuda or cpu)')
    return parser.parse_args()

def train_model(data_dir, save_path, epochs, batch_size, lr, device):
    os.makedirs(save_path, exist_ok=True)
    params_train = {'batch_size': batch_size, 'shuffle': True}
    params_validate = {'batch_size': batch_size, 'shuffle': True}

    x_folder = os.path.join(data_dir, 'x')
    x_list = os.listdir(x_folder)
    num_x = len(x_list)
    with open(os.path.join(data_dir, 'y.pickle'), 'rb') as handle:
        labels = pickle.load(handle)
    with open(os.path.join(data_dir, 'param.pickle'), 'rb') as handle:
        param_dict = pickle.load(handle)
    
    partition = {'train': x_list[:int(num_x*0.9)], 'validate': x_list[int(num_x*0.9):]}
    train_ds = MyDataset(x_folder, partition['train'], labels)
    train_dl = DataLoader(train_ds, **params_train)
    validate_ds = MyDataset(x_folder, partition['validate'], labels)
    validate_dl = DataLoader(validate_ds, **params_validate)

    D, us_factor, maxv = labels['volume_size'][0], labels['us_factor'], labels['blob_maxv']
    model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# of trainable parameters: {n_params}')

    optimizer = Adam(list(model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, min_lr=1e-6)
    if param_dict['us_factor']==1:
        my_loss_func = torch.nn.MSELoss()
    else:
        my_loss_func = KDE_loss3D(sigma=0.5*(param_dict['us_factor']/2), device=device)  # 0.5-2, 1.0-4
    trainer = TorchTrainer(model, my_loss_func, optimizer, lr_scheduler=scheduler, device=device)

    time_now = datetime.today().strftime('%m-%d_%H-%M')
    net_file = 'net_'+time_now+'.pt'
    checkpoints = dict(file_name=os.path.join(save_path, net_file),
                    net=Net(D=D, us_factor=us_factor, maxv=maxv),
                    state_dict=None,
                    note=' '
                    )

    t0 = time.time()
    fit_results = trainer.fit(train_dl, validate_dl, num_epochs=epochs, checkpoints=checkpoints, early_stopping=4)

    fit_file = 'fit_'+time_now+'.pickle'
    with open(os.path.join(save_path, fit_file), 'wb') as handle:
        pickle.dump(fit_results, handle)

    t1 = time.time()

    print(f'finished training in {t1-t0}s.')



if __name__ == "__main__":
    args = get_args()
    train_model(args.data_dir, args.save_path, args.epochs, args.batch_size, args.lr, args.device)
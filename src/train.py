import argparse
import os
import shutil
from efficientnet_pytorch import EfficientNet

from kekas import Keker, DataOwner
from kekas.metrics import bce_accuracy
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torchcontrib.optim import SWA
from tqdm import tqdm

from utils import read_config
from dataset import SteelDataset, EmptyMaskCallback, AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST
from loss import *
from metrics import SoftDiceCoef, HardDiceCoef, average_precision, roc_auc
from optimizer import RAdam
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Model(nn.Module):
    def __init__(self, base_model):
        super(Model, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            self.base_model,
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.classifier(x).squeeze()
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    config_main = read_config(args.config_file, "MAIN")
    config = read_config(args.config_file, "TRAIN")
    train_dataset = SteelDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TRAIN, phase='train')
    val_dataset = SteelDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TEST, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16)

    dataowner = DataOwner(train_loader, val_loader, None)

    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    if config['weight'] == "":
        model = EfficientNet.from_pretrained('efficientnet-b0')
    else:
        model = EfficientNet.from_name('efficientnet-b0')
        model_state = torch.load(config['weight'])

        new_model_state = OrderedDict()
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]
        model.load_state_dict(new_model_state)

    model = model.to(device)

    criterion = nn.BCELoss()

    if os.path.exists(os.path.join(config['log_path'], config['prefix'])):
        shutil.rmtree(os.path.join(config['log_path'], config['prefix']))
    else:
        os.makedirs(os.path.join(config['log_path'], config['prefix']))

    metrics = {"acc": bce_accuracy,
               "roc_auc": roc_auc,
               "average_precision": average_precision
               }

    # optimizer
    opt = RAdam if config['optimizer'] == 'radam' else Adam
    opt_params = {"weight_decay": config['weight_decay']}
    if config['swa'] > 0:
        print("Use SWA")
        opt_params = {"optimizer": opt(params=model.parameters(), weight_decay=config['weight_decay']),
                      "swa_start":config['n_epochs'] - config['swa'], "swa_freq":10, "swa_lr": 1e-5}
        opt = SWA

    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR
        sched_params = {"T_max": 10, "eta_min": 1e-6}
    else:
        scheduler = ExponentialLR
        sched_params = {"gamma": 0.9}

    keker = Keker(model=model,
                  dataowner=dataowner,
                  criterion=criterion,
                  target_key='label',
                  metrics=metrics,
                  opt=opt,
                  opt_params=opt_params,
                  device=device)

    if config['empty_mask_increase']['state'] == "true":
        empty_mask_callback = EmptyMaskCallback(start_value=config['empty_mask_increase']['start_value'],
                                                end_value=config['empty_mask_increase']['end_value'],
                                                n_epochs=config['empty_mask_increase']['n_epochs'])
        keker.add_callbacks([empty_mask_callback])

    keker.kek(lr=config['learning_rate'],
              epochs=config['n_epochs'],
              opt=opt,
              opt_params=opt_params,
              sched=scheduler,
              sched_params=sched_params,
              logdir=os.path.join(config['log_path'], config['prefix']),
              cp_saver_params={
                  "savedir": config['model_path'],
                  "metric": "hard_dice",
                  "n_best": 3,
                  "prefix": config['prefix'],
                  "mode": "max"}
              )

    keker.save(config['model_path'] + 'keker.kek')


if __name__ == '__main__':
    main()

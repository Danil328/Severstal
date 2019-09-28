import argparse
import os
import shutil
import segmentation_models_pytorch as smp

from kekas import Keker, DataOwner
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torchcontrib.optim import SWA
from tqdm import tqdm

from utils import read_config
from dataset import SteelDataset, AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST
from loss import *
from metrics import SoftDiceCoef, HardDiceCoef
from optimizer import RAdam
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--gpu-number", default="0,1", help="GPU for train", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    config_main = read_config(args.config_file, "MAIN")
    config = read_config(args.config_file, "TRAIN")
    train_dataset = SteelDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TRAIN, phase='train')
    val_dataset = SteelDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TEST, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16)

    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    if config['weight'] == "":
        model = smp.Unet('resnet34', encoder_weights='imagenet', activation=None, classes=4)
    else:
        model = smp.Unet('resnet34', encoder_weights=None, activation=None, classes=4)
        model_state = torch.load(config['weight'])

        new_model_state = OrderedDict()
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]
        model.load_state_dict(new_model_state)

    model = model.to(device)

    dataowner = DataOwner(train_loader, val_loader, None)
    criterion = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
    # criterion = FocalDiceLossWithoutLog(bce_weight=1.0, dice_weight=1.0)
    # criterion = lovasz_hinge()
    # criterion = FocalTverskyLossWithoutLog(alpha=0.5, beta=0.5, gamma=2.0, add_weight=False, pos_weight=2.0, neg_weight=1.0, bce_weight=1.0, dice_weight=1.0)

    if os.path.exists(os.path.join(config['log_path'], config['prefix'])):
        shutil.rmtree(os.path.join(config['log_path'], config['prefix']))
    else:
        os.makedirs(os.path.join(config['log_path'], config['prefix']))

    metrics = {"soft_dice": SoftDiceCoef(),
               "hard_dice": HardDiceCoef(threshold=0.5),
               "hard_dice_1": HardDiceCoef(class_id=0),
               "hard_dice_2": HardDiceCoef(class_id=1),
               "hard_dice_3": HardDiceCoef(class_id=2),
               "hard_dice_4": HardDiceCoef(class_id=3),
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
        sched_params = {"T_max": 8, "eta_min": 1e-6}
    else:
        scheduler = ExponentialLR
        sched_params = {"gamma": 0.9}

    keker = Keker(model=model,
                  dataowner=dataowner,
                  criterion=criterion,
                  target_key='mask',
                  metrics=metrics,
                  opt=opt,
                  opt_params=opt_params,
                  device=device)

    # keker.kek_lr(final_lr=0.1, logdir=config['log_path+"_find_lr")

    # keker.kek_one_cycle(max_lr=1e-4,  # the maximum learning rate
    #                     cycle_len=10,  # number of epochs, actually, but not exactly
    #                     momentum_range=(0.95, 0.85),  # range of momentum changes
    #                     div_factor=25,  # max_lr / min_lr
    #                     increase_fraction=0.3,
    #                     logdir=config['log_path'],
    #                     opt=torch.optim.Adam,
    #                     opt_params={"weight_decay": config['decay']},
    #                     cp_saver_params={
    #                         "savedir": config['model_path'],
    #                         "metric": "hard_dice",
    #                         "n_best": 3,
    #                         "prefix": config['prefix'],
    #                         "mode": "max"}
    #                     )

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

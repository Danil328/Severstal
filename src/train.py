import argparse
import os
import shutil
import segmentation_models_pytorch as smp

from kekas import Keker, DataOwner
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import read_config
from dataset import SteelDataset, AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST
from loss import *
from metrics import SoftDiceCoef, HardDiceCoef
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


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


    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if config['weight'] == "":
        model = smp.Unet('resnet34', encoder_weights='imagenet', activation=None, classes=4)
    else:
        model = smp.Unet('resnet34', encoder_weights=None, activation=None, classes=4)
        model_state = torch.load(config['weight'])

        new_model_state = OrderedDict()
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]
        model.load_state_dict(new_model_state)

    # model = R2AttU_Net()
    # init_weights(model)
    model = model.to(device)

    dataowner = DataOwner(train_loader, val_loader, None)
    criterion = BCEDiceLoss()
    # criterion = FocalDiceLossWithoutLog(bce_weight=1.0, dice_weight=1.0)
    # criterion = lovasz_hinge()
    # criterion = FocalTverskyLossWithoutLog(alpha=0.5, beta=0.5, gamma=2.0, add_weight=False, pos_weight=2.0, neg_weight=1.0, bce_weight=1.0, dice_weight=1.0)

    if os.path.exists(config['log_path']):
        shutil.rmtree(config['log_path'])
    else:
        os.makedirs(config['log_path'])

    # opt = torch.optim.Adam(model.parameters())
    # for batch in tqdm(train_loader):
    #     opt.zero_grad()
    #     image = batch['image'].to(device)
    #     mask = batch['mask'].to(device)
    #     output = model(image)
    #     loss = criterion(output, mask)
    #     loss.backward()
    #     opt.step()

    keker = Keker(model=model,
                  dataowner=dataowner,
                  criterion=criterion,
                  target_key='mask',
                  metrics={"soft_dice": SoftDiceCoef(), "hard_dice": HardDiceCoef(threshold=0.5)},
                  opt=torch.optim.Adam,
                  opt_params={"weight_decay": config['weight_decay']},
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
              opt=torch.optim.Adam,
              opt_params={"weight_decay": config['weight_decay']},
              sched=torch.optim.lr_scheduler.ExponentialLR,
              sched_params={"gamma": 0.95},
              # sched=torch.optim.lr_scheduler.CosineAnnealingLR,
              # sched_params={"T_max": 10, "eta_min": 5e-5},
              # sched=torch.optim.lr_scheduler.CyclicLR,
              # sched_params={"base_lr": 1e-5, "max_lr": 1e-4, "step_size_up": 3, "step_size_down": 3, "mode": "exp_range",
              #               "gamma": 0.95, "base_momentum": 0.8, "max_momentum": 0.95},              # sched=torch.optim.lr_scheduler.CyclicLR,
              logdir=config['log_path'],
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

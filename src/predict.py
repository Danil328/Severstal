import argparse
import os
import shutil
import segmentation_models_pytorch as smp

from kekas import Keker, DataOwner
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torchcontrib.optim import SWA
import numpy as np
from tqdm import tqdm

from utils import read_config
from dataset import SteelDataset, AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST, AUGMENTATIONS_TEST_FLIPPED
from metrics import dice_coef_numpy
from loss import *
from metrics import SoftDiceCoef, HardDiceCoef
from optimizer import RAdam
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    config_main = read_config(args.config_file, "MAIN")
    config = read_config(args.config_file, "TEST")
    val_dataset = SteelDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TEST, phase='val')
    test_dataset = SteelDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TEST, phase='test')
    test_dataset_flip = SteelDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TEST_FLIPPED, phase='test')

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16)
    test_loader_flip = DataLoader(test_dataset_flip, batch_size=config['batch_size'], shuffle=False, num_workers=16)

    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    model = smp.Unet('resnet34', encoder_weights=None, activation=None, classes=4)
    model_state = torch.load(config['weight'])

    new_model_state = OrderedDict()
    for key in model_state.keys():
        new_model_state[key[7:]] = model_state[key]
    model.load_state_dict(new_model_state)
    model = model.to(device)
    model.eval()

    best_threshold, best_noise_threshold = search_threshold(model, val_loader, device)


def search_threshold(model, val_loader, device):
    masks, predicts = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            image = batch['image'].to(device)
            mask = batch['mask'].cpu().numpy()
            predict = model(image).cpu().numpy()
            masks.append(mask)
            predicts.append(predict)

    # TODO convert to array

    # Search threshold
    thresholds = np.arange(0.1, 0.9, 0.5)
    scores = []
    for threshold in thresholds:
        scores.append(dice_coef_numpy(preds=(predicts>threshold).astype(int), trues=masks))
    best_score = np.max(scores)
    best_threshold = thresholds[np.argmax(scores)]
    print(f"Best threshold - {best_threshold}, best score - {best_score}")

    # Search noise threshold
    predicts = (predicts>best_threshold).astype(int)
    thresholds = np.arange(100, 1000, 100)
    scores = []
    for threshold in thresholds:
        scores.append(dice_coef_numpy(preds=predicts, trues=masks, noise_threshold=threshold))
    best_score = np.max(scores)
    best_noise_threshold = thresholds[np.argmax(scores)]
    print(f"Best noise threshold - {best_noise_threshold}, best score - {best_score}")

    return best_threshold, best_noise_threshold

def predict(model, test_loader, test_loader_flip, device):
    predicts, predicts_flip = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            image = batch['image'].to(device)
            predict = model(image).cpu().numpy()
            predicts.append(predict)

    with torch.no_grad():
        for batch in tqdm(test_loader_flip):
            image = batch['image'].to(device)
            predict = model(image).cpu().numpy()
            predicts_flip.append(predict)

    # TODO convert to array

    # TODO averaging

    # TODO apply thresholds





if __name__ == '__main__':
    main()
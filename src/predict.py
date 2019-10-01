import argparse
import os
import pydoc

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SteelDataset, AUGMENTATIONS_TEST, AUGMENTATIONS_TEST_FLIPPED
from metrics import dice_coef_numpy
from models.unet import ResnetSuperVision
from utils import read_config

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

    # model = pydoc.locate(config['model'])(**config['model_params'])
    model = ResnetSuperVision(**config['model_params'])
    if isinstance(config.get('weights', None), str):
        model.load_state_dict(torch.load(config['weights']))
    model = model.to(device)
    model.eval()

    # best_threshold, best_noise_threshold = search_threshold(model, val_loader, device)
    best_threshold = 0.8
    best_noise_threshold = 800

    predicts = predict(model, test_loader, test_loader_flip, device)
    predicts = apply_thresholds(predicts, best_threshold, best_noise_threshold)


def search_threshold(model, val_loader, device):
    masks, predicts = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            image = batch['image'].to(device)
            mask = batch['mask'].cpu().numpy()
            predict_mask, predict_label = model(image)
            masks.append(mask)
            predicts.append(predict_mask.cpu().numpy())

    predicts = np.vstack(predicts)
    masks = np.vstack(masks)

    print("Search threshold ...")
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = []
    for threshold in tqdm(thresholds):
        score = dice_coef_numpy(preds=(predicts>threshold).astype(int), trues=masks)
        scores.append(score)
    best_score = np.max(scores)
    best_threshold = thresholds[np.argmax(scores)]
    print(f"Best threshold - {best_threshold}, best score - {best_score}")

    print("Search noise threshold ...")
    predicts = (predicts>best_threshold).astype(int)
    thresholds = np.arange(100, 1000, 100)
    scores = []
    for threshold in tqdm(thresholds):
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
            predict_mask, predict_label = model(image)
            predicts.append(predict_mask.cpu().numpy())

    with torch.no_grad():
        for batch in tqdm(test_loader_flip):
            image = batch['image'].to(device)
            predict_mask, predict_label = model(image)
            predicts_flip.append(predict_mask.cpu().numpy())

    predicts = np.vstack(predicts)
    predicts_flip = np.vstack(predicts_flip)

    predicts = predicts * 0.5 + predicts_flip * 0.5
    return predicts


def apply_thresholds(predicts: np.ndarray, threshold: float, noise_threshold: float):
    predicts = (predicts>threshold).astype(int)
    # predicts[predicts.sum(1) < noise_threshold, ...] = 0
    return predicts





if __name__ == '__main__':
    main()
import argparse
import glob
import os
import pydoc

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SteelDataset, AUGMENTATIONS_TEST, AUGMENTATIONS_TEST_FLIPPED
from metrics import dice_coef_numpy
from models.unet import ResnetSuperVision
from utils import read_config, mask2rle
import segmentation_models_pytorch as smp

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

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16, drop_last=False)
    test_loader_flip = DataLoader(test_dataset_flip, batch_size=config['batch_size'], shuffle=False, num_workers=16, drop_last=False)

    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    # best_threshold, best_min_size_threshold = search_threshold(config, val_loader, device)
    best_threshold = 0.85
    best_min_size_threshold = 600

    predict(config, test_loader, best_threshold, best_min_size_threshold, device)


def search_threshold(config, val_loader, device):
    models = []

    for weight in glob.glob(os.path.join(config['weights'], config['name'], 'cosine/') + "*.pth"):
        if 'smp' in config['model']:
            model = smp.Unet(encoder_name=config['model_params']['backbone_arch'],
                             encoder_weights=None,
                             classes=config['model_params']['seg_classes'])
        else:
            model = ResnetSuperVision(**config['model_params'])
        model.load_state_dict(torch.load(weight))
        model = model.to(device)
        model.eval()
        models.append(model)

    masks, predicts = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            images = batch["image"].to(device)
            mask = batch['mask'].cpu().numpy()
            batch_preds = np.zeros((images.size(0), 4, 256, 1600), dtype=np.float32)
            if config['type'] == 'crop':
                for model in models:
                    tmp_batch_preds = np.zeros((images.size(0), 4, 256, 1600), dtype=np.float32)
                    for step in np.arange(0, 1600, 384)[:-1]:
                        tmp_pred = torch.sigmoid(model(images[:,:,:,step:step+448])).cpu().numpy()
                        tmp_batch_preds[:,:,:,step:step+448] += tmp_pred
                    tmp_batch_preds[:,:,:,384:384+64] /= 2
                    tmp_batch_preds[:,:,:,2*384:2*384+64] /= 2
                    tmp_batch_preds[:,:,:,3*384:3*384+64] /= 2
                    batch_preds += tmp_batch_preds
            else:
                for model in models:
                    batch_preds += torch.sigmoid(model(images)).cpu().numpy()
            batch_preds = batch_preds / len(models)

            masks.append(mask)
            predicts.append(batch_preds)

    predicts = np.vstack(predicts)
    masks = np.vstack(masks)

    print("Search threshold ...")
    thresholds = np.arange(0.25, 0.9, 0.05)
    scores = []
    for threshold in tqdm(thresholds):
        score = dice_coef_numpy(preds=(predicts>threshold).astype(int), trues=masks)
        print(score)
        scores.append(score)
    best_score = np.max(scores)
    best_threshold = thresholds[np.argmax(scores)]
    print(f"Best threshold - {best_threshold}, best score - {best_score}")
    print(f"Scores: {scores}")

    print("Search min_size threshold ...")
    predicts = (predicts > best_threshold).astype(np.uint8)
    thresholds = np.arange(100, 4000, 500)
    scores = []
    for threshold in tqdm(thresholds):
        tmp = predicts.copy()
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i,j] = post_process(tmp[i,j], best_threshold, threshold, isVal=True)[0]
        score = dice_coef_numpy(preds=tmp, trues=masks)
        print(score)
        scores.append(score)
    best_score = np.max(scores)
    best_min_size_threshold = thresholds[np.argmax(scores)]
    print(f"Best min_size threshold - {best_min_size_threshold}, best score - {best_score}")
    print(f"Scores: {scores}")

    return best_threshold, best_min_size_threshold


def predict(config, test_loader, best_threshold, min_size, device):
    models = []
    for weight in glob.glob(os.path.join(config['weights'], config['name'], 'cosine/') + "*.pth"):
        if 'smp' in config['model']:
            model = smp.Unet(encoder_name=config['model_params']['backbone_arch'],
                             encoder_weights=None,
                             classes=config['model_params']['seg_classes'])
        else:
            model = ResnetSuperVision(**config['model_params'])
        model.load_state_dict(torch.load(weight))
        model = model.to(device)
        model.eval()
        models.append(model)

    predictions = []
    image_names = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            fnames = batch["filename"]
            images = batch["image"].to(device)
            batch_preds = np.zeros((images.size(0), 4, 256, 1600), dtype=np.float32)
            if config['type'] == 'crop':
                for model in models:
                    tmp_batch_preds = np.zeros((images.size(0), 4, 256, 1600), dtype=np.float32)
                    for step in np.arange(0, 1600, 384)[:-1]:
                        tmp_pred = torch.sigmoid(model(images[:, :, :, step:step + 448])).cpu().numpy()
                        tmp_batch_preds[:, :, :, step:step + 448] += tmp_pred
                    tmp_batch_preds[:, :, :, 384:384 + 64] /= 2
                    tmp_batch_preds[:, :, :, 2 * 384:2 * 384 + 64] /= 2
                    tmp_batch_preds[:, :, :, 3 * 384:3 * 384 + 64] /= 2
                    batch_preds += tmp_batch_preds
            else:
                for model in models:
                    batch_preds += torch.sigmoid(model(images)).cpu().numpy()
            batch_preds = batch_preds / len(models)
            for fname, preds in zip(fnames, batch_preds):
                for cls, pred in enumerate(preds):
                    pred, num = post_process(pred, best_threshold, min_size)
                    rle = mask2rle(pred)
                    name = fname + f"_{cls + 1}"
                    image_names.append(name)
                    predictions.append(rle)

    df = pd.DataFrame()
    df["ImageId_ClassId"] = image_names
    df["EncodedPixels"] = predictions
    df.to_csv(os.path.join(config['weights'], config['name'], "submission.csv"), index=False)


def post_process(mask, threshold, min_size, isVal=False):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    if not isVal:
        mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


if __name__ == '__main__':
    main()
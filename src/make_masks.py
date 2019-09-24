import argparse

import cv2
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import read_config, rle2mask


def parse_args():
    parser = argparse.ArgumentParser(description="Create mask for training")
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config_file, stage="MAIN")

    if not os.path.exists(os.path.join(config['path_to_data'], 'train_masks')):
        os.makedirs(os.path.join(config['path_to_data'], 'train_masks'))

    df = pd.read_csv(os.path.join(config['path_to_data'], 'train.csv'))
    df['ImageId'] = df['ImageId_ClassId'].map(lambda x: x.split("_")[0])
    df['ClassId'] = df[['ImageId_ClassId', 'EncodedPixels']].apply(
        lambda x: x[0].split("_")[1] if x[1] is not np.nan else '0', axis=1)
    df.drop(columns='ImageId_ClassId', inplace=True)
    df.drop_duplicates(inplace=True)

    # drop row with label 0 which exist label != 0
    max_class_df = df.groupby("ImageId")['ClassId'].max().reset_index()
    max_class_df = max_class_df[max_class_df['ClassId'] != '0']
    df = df[~((df['ImageId'].isin(max_class_df['ImageId'].values)) & (df['ClassId'] == '0'))]

    # split to train and validation
    # Didn't use
    # tv_df = df.groupby("ImageId")['ClassId'].sum().reset_index()
    # train_images, val_images = train_test_split(tv_df['ImageId'].values, stratify=tv_df['ClassId'].values, test_size=0.2)

    images = df['ImageId'].unique()
    for image in tqdm(images):
        temp_df = df[df['ImageId'] == image]
        mask = np.zeros((4, 256, 1600), dtype=np.uint8)
        for row in temp_df.values:
            if row[0] is not np.nan:
                mask[int(row[2]) - 1] = rle2mask(row[0])
        mask = np.moveaxis(mask, 0, -1)
        cv2.imwrite(os.path.join(config['path_to_data'], 'train_masks', image.replace('.jpg', '.png')), mask)


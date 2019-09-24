import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (
    HorizontalFlip,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    Normalize,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Blur, RandomContrast, RandomBrightness)
from albumentations.pytorch import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from torch.utils.data import Dataset

ORIG_SHAPE = (256, 1600)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomGamma(),
        RandomBrightnessContrast(),
    ], p=0.2),
    # OneOf([
    #     ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #     GridDistortion(),
    #     OpticalDistortion(distort_limit=2, shift_limit=0.5),
    # ], p=0.3),
    OneOf([
        IAAAdditiveGaussianNoise(),
        GaussNoise(),
    ], p=0.1),
    OneOf([
        MotionBlur(blur_limit=5, p=0.2),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),
    ], p=0.1),
    OneOf([
        CLAHE(clip_limit=2),
        RandomBrightnessContrast(),
    ], p=0.1),
    Normalize(),
    ToTensor(num_classes=4, sigmoid=True)
], p=1)

AUGMENTATIONS_TEST = Compose([
    Normalize(),
    ToTensor(num_classes=4, sigmoid=True)
], p=1)


class SteelDataset(Dataset):
    def __init__(self, data_folder, transforms, phase):
        assert phase in ['train', 'val', 'test'], "Fuck you!"

        self.root = data_folder
        self.transforms = transforms
        self.phase = phase
        if phase != 'test':
            self.images = self.split_train_val(glob.glob(os.path.join(self.root, "train_images", "*.jpg")))
        else:
            self.images = glob.glob(os.path.join(self.root, "test_images", "*.jpg"))

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        mask = cv2.imread(self.images[idx].replace('train_images', 'train_masks').replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        return {"image": img, "mask": mask}

    def __len__(self):
        return len(self.images)

    def split_train_val(self, images: list):
        train, val = train_test_split(images, test_size=0.2, random_state=17)
        if self.phase == 'train':
            return train
        elif self.phase == 'val':
            return val


if __name__ == '__main__':
    rnd = np.random.randint(1, 100)
    dataset = SteelDataset(data_folder='../data', transforms=AUGMENTATIONS_TRAIN, phase='train')
    data = dataset[rnd]
    image = data['image'].numpy()
    mask = data['mask'].numpy()
    print(image.shape)
    print(mask.shape)
    print(image.min(), image.max())
    print(mask.min(), mask.max())

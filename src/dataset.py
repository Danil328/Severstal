import glob
import os
import random

import cv2
import numpy as np
import torch
from albumentations import (
    HorizontalFlip,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    OneOf,
    RandomBrightnessContrast,
    Normalize,
    ShiftScaleRotate,
    CropNonEmptyMaskIfExists)
from albumentations.pytorch import ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

ORIG_SHAPE = (256, 1600)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2), rotate_limit=(-20, 20), border_mode=0, interpolation=1, p=0.25),
    OneOf([
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    ], p=0.25),
    # OneOf([
    #     ElasticTransform(alpha=120, alpha_affine=120 * 0.03, approximate=False, border_mode=0, interpolation=1, sigma=6, p=0.5),
    #     GridDistortion(border_mode=0, distort_limit=(-0.3, 0.3), interpolation=1, num_steps=5, p=0.5),
    #     OpticalDistortion(border_mode=0, distort_limit=(-2, 2), interpolation=1, shift_limit=(-0.5, 0.5), p=0.5),
    # ], p=0.3),
    Normalize(),
    ToTensor(num_classes=4, sigmoid=True)
], p=1)

AUGMENTATIONS_TEST = Compose([
    Normalize(),
    ToTensor(num_classes=4, sigmoid=True)
], p=1)

AUGMENTATIONS_TEST_FLIPPED = Compose([
    HorizontalFlip(p=1.0, always_apply=True),
    Normalize(),
    ToTensor(num_classes=4, sigmoid=True)
], p=1)


AUGMENTATIONS_TRAIN_CROP = Compose([
    CropNonEmptyMaskIfExists(height=256, width=448, always_apply=True),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2), rotate_limit=(-20, 20), border_mode=0, interpolation=1, p=0.25),
    OneOf([
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    ], p=0.25),
    Normalize(),
    ToTensor(num_classes=4, sigmoid=True)
], p=1)

AUGMENTATIONS_TEST_CROP = Compose([
    CropNonEmptyMaskIfExists(height=256, width=448, always_apply=True),
    Normalize(),
    ToTensor(num_classes=4, sigmoid=True)
], p=1)

class SteelDataset(Dataset):
    """
    0 - empty mask
    Class balance
    0       5902
    1       769
    12       35
    123       2
    13       91
    2       195
    23       14
    24        1
    3      4759
    34      284
    4       516
    """
    def __init__(self, data_folder, transforms, phase, empty_mask_params: dict = None):
        assert phase in ['train', 'val', 'test'], "Fuck you!"

        self.root = data_folder
        self.transforms = transforms
        self.phase = phase
        if phase != 'test':
            self.images = np.asarray(self.split_train_val(glob.glob(os.path.join(self.root, "train_images", "*.jpg"))))[:200]

            # Get labels for classification
            self.labels = np.zeros((self.images.shape[0], 4), dtype=np.float32)
            for idx, image_name in enumerate(tqdm(self.images)):
                image = cv2.imread(image_name.replace('train_images', 'train_masks').replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)
                self.labels[idx] = (np.amax(image, axis=(0, 1)) > 0).astype(float)

            self.empty_images = self.images[self.labels.max(axis=1) == 0]
            self.non_empty_images = self.images[self.labels.max(axis=1) == 1]
        else:
            self.images = glob.glob(os.path.join(self.root, "test_images", "*.jpg"))#[:200]

        if empty_mask_params is not None and empty_mask_params['state'] == 'true':
            self.start_value = empty_mask_params['start_value']
            self.delta = (empty_mask_params['end_value'] - empty_mask_params['start_value']) / empty_mask_params['n_epochs']
            self.positive_ratio = self.start_value
        else:
            self.positive_ratio = 1.0


    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        if self.phase != 'test':
            mask = cv2.imread(self.images[idx].replace('train_images', 'train_masks').replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            return {"image": img, "mask": mask, "label": mask.max()} #torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            augmented = self.transforms(image=img)
            img = augmented['image']
            return {"image": img, "filename": self.images[idx].split("/")[-1]}

    def __len__(self):
        return len(self.images)

    def split_train_val(self, images: list):
        train, val = train_test_split(images, test_size=0.1, random_state=17)
        if self.phase == 'train':
            return train
        elif self.phase == 'val':
            return val

    def update_empty_mask_ratio(self, epoch: int):
        self.positive_ratio = self.start_value + self.delta * epoch
        self.images = np.hstack((self.non_empty_images, self.empty_images[:int(self.positive_ratio * self.empty_images.shape[0])]))


class FourBalanceClassSampler(Sampler):
    def __init__(self, labels):
        label = labels.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        #assume we know neg is majority class
        num_neg = len(self.neg_index)
        self.length = 4*num_neg


    def __iter__(self):
        neg = self.neg_index.copy()
        random.shuffle(neg)
        num_neg = len(self.neg_index)

        pos1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    rnd = np.random.randint(1, 100)
    dataset = SteelDataset(data_folder='../data', transforms=AUGMENTATIONS_TRAIN, phase='train',
                           empty_mask_params={"state": "true",
                                              "start_value": 0.0,
                                              "end_value": 1.0,
                                              "n_epochs": 50})
    data = dataset[rnd]
    image = data['image'].numpy()
    mask = data['mask'].numpy()
    label = data['label'].numpy()
    print(image.shape)
    print(mask.shape)
    print(label.shape)
    print(image.min(), image.max())
    print(mask.min(), mask.max())
    print(dataset.labels.sum(0))

    print(f"Len before update {dataset.__len__()}")
    dataset.update_empty_mask_ratio(10)
    print(f"Len after update {dataset.__len__()}")


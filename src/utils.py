import random

import numpy as np
import torch
from torch import nn
import torch.functional as F
import yaml


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600, 256)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape).T


def read_config(path, stage:str):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)[stage]
        except yaml.YAMLError as exc:
            print(exc)
            return None


def set_global_seeds(i: int):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)


def batch2device(data, device):
    return {k: v if not hasattr(v, 'to') else v.to(device) for k, v in data.items()}


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))
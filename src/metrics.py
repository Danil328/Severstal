import numpy as np
import torch
from torch import nn


# Segmentation
def dice_coef(preds, trues, smooth=1e-3):
    preds = preds.contiguous().view(preds.size(0), -1).float()
    trues = trues.contiguous().view(preds.size(0), -1).float()
    inter = torch.sum(preds * trues, dim=1)
    dice = torch.mean((2.0 * inter + smooth) / (preds.sum(dim=1) + trues.sum(dim=1) + smooth))
    return dice


def dice_coef_numpy(preds, trues, smooth=1e-3, noise_threshold=0):
    preds = preds.reshape(preds.shape[0], -1)
    trues = trues.reshape(trues.shape[0], -1)

    if noise_threshold > 0:
        preds[preds.sum(1) < noise_threshold, ...] = 0

    inter = np.sum(preds * trues, 1)
    dice = np.mean((2.0 * inter + smooth) / (preds.sum(1) + trues.sum(1) + smooth))
    return dice


class SoftDiceCoef(nn.Module):
    def __init__(self, class_id=-1):
        self.class_id = class_id
        super().__init__()

    def forward(self, inputs, target):
        if self.class_id > -1:
            inputs = inputs[:, self.class_id]
            target = target[:, self.class_id]
        inputs = torch.sigmoid(inputs)
        dice = dice_coef(inputs, target)
        return dice


class HardDiceCoef(nn.Module):
    def __init__(self, class_id=None, threshold=0.5):
        super().__init__()
        self.class_id = class_id
        self.threshold = threshold

    def forward(self, inputs, target):
        if self.class_id is not None:
            inputs = inputs[:, self.class_id]
            target = target[:, self.class_id]
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        dice = dice_coef(inputs, target)
        return dice
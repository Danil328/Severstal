import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
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
    def __init__(self, class_id=None):
        self.class_id = class_id
        super().__init__()

    def forward(self, inputs, target):
        if self.class_id is not None:
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


def average_precision(target: torch.Tensor,
                 preds: torch.Tensor) -> float:
    if target.max() == 0:
        target = torch.cat((target, torch.tensor([1.0], device='cuda')))
        preds = torch.cat((preds, torch.tensor([1.0], device='cuda')))
    if target.min() == 1:
        target = torch.cat((target, torch.tensor([0.0], device='cuda')))
        preds = torch.cat((preds, torch.tensor([0.0], device='cuda')))
    target = target.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    return average_precision_score(target, preds)


def roc_auc(target: torch.Tensor,
            preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    return roc_auc_score(target, preds)
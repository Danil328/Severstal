import torch
import torch.nn.functional as F
from torch import nn

from losses.my_loss import TverskyLoss
from .bce import StableBCELoss
from .dice import DiceLoss
from .focal import FocalLoss2d
from .jaccard import JaccardLoss
from .lovasz import LovaszLoss, LovaszLossSigmoid


class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False, channel_weights=None, channel_losses=None):
        super().__init__()
        if channel_weights is None:
            channel_weights = [1, 1, 1, 1]
        self.weights = weights

        # https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/110188#latest-637863
        self.bce = nn.BCELoss(weight=torch.tensor([1.0, 0.75])) #StableBCELoss()

        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.lovasz = LovaszLoss(per_image=per_image)
        self.lovasz_sigmoid = LovaszLossSigmoid(per_image=per_image)
        self.focal = FocalLoss2d()
        self.tversky = TverskyLoss(alpha=0.5, beta=0.5, gamma=2.0)
        self.mapping = {
            'bce': self.bce,
            'dice': self.dice,
            'focal': self.focal,
            'jaccard': self.jaccard,
            'lovasz': self.lovasz,
            'lovasz_sigmoid': self.lovasz_sigmoid,
            'tversky': self.tversky
        }
        self.expect_sigmoid = {'dice', 'focal', 'jaccard', 'lovasz_sigmoid', 'tversky', 'bce'}
        self.per_channel = {'dice', 'jaccard', 'lovasz_sigmoid', 'tversky'}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = targets.size(1)
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        val += self.channel_weights[c] * self.mapping[k](
                            sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
                            targets[:, c, ...]
                        )
            else:
                val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)

            self.values[k] = val
            loss += self.weights[k] * val
        return loss.clamp(min=1e-5)


class ComboSuperVisionLoss(ComboLoss):
    def __init__(self, weights, per_image=False, channel_weights=(1.0, 1.0, 1.0, 1.0), channel_losses=None, sv_weight=0.15):
        super().__init__(weights, per_image, channel_weights, channel_losses)
        self.sv_weight = sv_weight

    def forward(self, *input):
        outputs, targets, sv_outputs, sv_targets = input
        mask_loss = super().forward(outputs, targets)
        supervision_loss = F.binary_cross_entropy_with_logits(sv_outputs, sv_targets)
        return self.sv_weight * supervision_loss + (1 - self.sv_weight) * mask_loss


import math

import torch
import torch.nn as nn


class WingLoss(nn.Module):
    def __init__(self, omega, epsilon, emoji_weights=None):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.emoji_weights = emoji_weights
        if emoji_weights is not None:
            self.emoji_weights = torch.tensor(emoji_weights, dtype=torch.float32).cuda()

    def forward(self, pred, target, mask=None):
        """
        pred: outputs['mouth_emoji'], [B, N, 46];
        target: inputs['mouth_emoji'], [B, N, 46];
        mask: [B, N, 1]
        """
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        if self.emoji_weights is not None:
            delta_y = delta_y * self.emoji_weights
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]

        if mask is not None:
            mask = mask.unsqueeze(2).expand(y.size())
            mask1 = mask[delta_y < self.omega]
            mask2 = mask[delta_y >= self.omega]
            delta_y1 = delta_y1[mask1]
            delta_y2 = delta_y2[mask2]

        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

import math
from dataclasses import dataclass


import torch
import torch.nn as nn


@dataclass
class WingLossConfig:
    omega: float = 10.
    epsilon: float = 2.0
    emoji_weight: float = 1.0


@dataclass
class LossConfig:
    use_wing_loss: bool = False
    use_l1_loss: bool = True

    wing_loss_weight: float = 1.
    l1_loss_weight: bool = 1.

    wing_loss_config: WingLossConfig = WingLossConfig()
    

class WingLoss(nn.Module):
    def __init__(self, omega, epsilon, emoji_weight=None, device=None):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.emoji_weight = emoji_weight
        # if emoji_weight is not None:
            # self.emoji_weight = torch.tensor(emoji_weight, dtype=torch.float32).to(device)

    def forward(self, pred, target, mask=None):
        """
        pred: outputs['mouth_emoji'], [B, N, 46];
        target: inputs['mouth_emoji'], [B, N, 46];
        mask: [B, N, 1]
        """
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        if self.emoji_weight is not None:
            delta_y = delta_y * self.emoji_weight

        lower_thres_mask = (delta_y < self.omega)
        higher_thres_mask = (delta_y >= self.omega)
        
        delta_y1 = delta_y[lower_thres_mask]
        delta_y2 = delta_y[higher_thres_mask]

        if mask is not None:
            mask = mask.unsqueeze(2).expand(y.size())
            mask1 = mask[lower_thres_mask]
            mask2 = mask[higher_thres_mask]
            delta_y1 = delta_y1[mask1]
            delta_y2 = delta_y2[mask2]

        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        # loss2 = delta_y2

        record_loss = (loss1.sum() + loss2.sum()).item()
        record_num = (len(loss1) + len(loss2))

        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2)), record_loss, record_num
        # return loss1.sum() / len(loss1) + loss2.sum() / len(loss2), record_loss, record_num


class WingFlattenLoss:
    """
    直接对 flatten 后并且手动去除的预测和标签计算损失函数；

    pred: [M, H]
    target: [M, H]
    
    """
    def __init__(self, omega, epsilon, emoji_weight=None):
        self.omega = omega
        self.epsilon = epsilon
        self.emoji_weight = emoji_weight
        # if emoji_weight is not None:
            # self.emoji_weight = torch.tensor(emoji_weight, dtype=torch.float32).to(device)

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        if self.emoji_weight is not None:
            delta_y = delta_y * self.emoji_weight
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]

        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C

        record_loss = (loss1.sum() + loss2.sum()).item()
        record_num = (len(loss1) + len(loss2))

        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2)), record_loss, record_num
    


class L1Loss:
    def __init__(self) -> None:
        ...

    def __call__(self, pred, target, mask=None):

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        if mask is not None:
            mask = mask.unsqueeze(2).expand(y.size())
            delta_y = delta_y[mask]

        loss = delta_y.sum()
        record_loss = loss.item()
        record_num = len(delta_y)

        loss = loss / record_num
        return loss, record_loss, record_num
    

class L1FlattenLoss:
    def __init__(self) -> None:
        ...

    def __call__(self, pred, target):

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()

        loss = delta_y.sum()
        record_loss = loss.item()
        record_num = delta_y.numel()

        loss = loss / record_num
        return loss, record_loss, record_num
    

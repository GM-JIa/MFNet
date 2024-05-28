import torch
import torch.nn as nn

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, mask):
        outputs = outputs.view(-1)
        mask = mask.view(-1)
        intersection = (outputs * mask).sum()
        dice_coefficient = (2.0 * intersection + self.smooth) / (outputs.sum() + mask.sum() + self.smooth)
        loss = 1 - dice_coefficient
        return loss

class NetLoss(nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()
        self.weight = nn.Parameter(torch.tensor(0.5),requires_grad=True)
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, outputs, mask):
        dice_loss = self.dice_loss(outputs, mask)
        bce_loss = self.bce_loss(outputs, mask)


        combined_loss = self.weight * bce_loss + (1-self.weight) * dice_loss
        return combined_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # class weights, same as CrossEntropyLoss weight

    def forward(self, inputs, targets):
        # standard cross entropy per sample
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')

        # probability of the true class
        pt = torch.exp(-ce_loss)

        # focal modulation — down-weights easy examples
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()
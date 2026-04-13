# Model/naive_resnet18.py

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class NaiveResNet18(nn.Module):
    def __init__(self, n_classes=5, dropout=0.4, pretrained=True):
        super().__init__()
        weights    = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone   = resnet18(weights=weights)

        # keep everything except the final fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
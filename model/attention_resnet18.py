import torch
import torch.nn as nn
from Module.residual_unit import ResidualUnit
from torchvision.models import resnet18, ResNet18_Weights
from Module.attention_block import AttentionBlock

class AttentionResNet18(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, dropout=0.4, pretrained=False):
        super().__init__()

        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )   # 135x135 -> 34x34

        # Stage 2
        self.stage2 = nn.Sequential(
            ResidualUnit(64, 64, stride=1),
            ResidualUnit(64, 64, stride=1),
        )
        self.attn2 = AttentionBlock(64, skip=2)

        # Stage 3
        self.stage3 = nn.Sequential(
            ResidualUnit(64, 128, stride=2),
            ResidualUnit(128, 128, stride=1),
        )
        self.attn3 = AttentionBlock(128, skip=1)

        # Stage 4
        self.stage4 = nn.Sequential(
            ResidualUnit(128, 256, stride=2),
            ResidualUnit(256, 256, stride=1),
        )
        self.attn4 = AttentionBlock(256, skip=1)

        # Stage 5 — spatial too small for attention
        self.stage5 = nn.Sequential(
            ResidualUnit(256, 512, stride=2),
            ResidualUnit(512, 512, stride=1),
        )

        # Head
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(512, n_classes)

        if pretrained:
            self._load_imagenet_weights()
    
    def _load_imagenet_weights(self):
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Stage 1: conv1 + bn1 + relu + maxpool
        self.stage1[0].load_state_dict(backbone.conv1.state_dict())
        self.stage1[1].load_state_dict(backbone.bn1.state_dict())

        # Stages 2-5: each layer in torchvision resnet18 maps to your stages
        # backbone.layer1 = stage2, layer2 = stage3, layer3 = stage4, layer4 = stage5
        for our_stage, their_layer in [
            (self.stage2, backbone.layer1),
            (self.stage3, backbone.layer2),
            (self.stage4, backbone.layer3),
            (self.stage5, backbone.layer4),
        ]:
            for our_block, their_block in zip(our_stage, their_layer):
                our_block.conv1.load_state_dict(their_block.conv1.state_dict())
                our_block.conv2.load_state_dict(their_block.conv2.state_dict())
                our_block.bn1.load_state_dict(their_block.bn1.state_dict())
                our_block.bn2.load_state_dict(their_block.bn2.state_dict())
                if hasattr(their_block, 'downsample') and their_block.downsample is not None:
                    our_block.shortcut.load_state_dict(their_block.downsample[0].state_dict())

        print("Loaded ImageNet weights into backbone stages")

    def forward(self, x):
        x = self.stage1(x)
        x = self.attn2(self.stage2(x))
        x = self.attn3(self.stage3(x))
        x = self.attn4(self.stage4(x))
        x = self.stage5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
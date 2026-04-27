import warnings

import torch
import torch.nn as nn

from Module.attention_block import AttentionBlock
from Module.residual_unit import ResidualUnit


class CustomCNNResidualAttention(nn.Module):
    def __init__(
        self,
        n_classes=5,
        dropout=0.4,
        pretrained=False,
        base_channels=64,
        adaptive_pool_size=8,
        classifier_hidden_dim=1024,
        classifier_bottleneck_dim=256,
        residual_depth=2,
        use_attention=True,
        attention_skip=1,
    ):
        super().__init__()

        if pretrained:
            warnings.warn(
                "CustomCNNResidualAttention does not support pretrained weights. Training from scratch.",
                RuntimeWarning,
                stacklevel=2,
            )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c1, c1, residual_depth)
        self.stage2 = self._make_stage(c1, c2, residual_depth)
        self.stage3 = self._make_stage(c2, c3, residual_depth)
        self.stage4 = self._make_stage(c3, c4, residual_depth)

        if use_attention:
            self.attn1 = AttentionBlock(c1, skip=attention_skip)
            self.attn2 = AttentionBlock(c2, skip=attention_skip)
            self.attn3 = AttentionBlock(c3, skip=attention_skip)
            self.attn4 = AttentionBlock(c4, skip=attention_skip)
        else:
            self.attn1 = nn.Identity()
            self.attn2 = nn.Identity()
            self.attn3 = nn.Identity()
            self.attn4 = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size))

        flattened_dim = c4 * adaptive_pool_size * adaptive_pool_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_bottleneck_dim),
            nn.BatchNorm1d(classifier_bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_bottleneck_dim, n_classes),
        )

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, residual_depth):
        blocks = [ResidualUnit(in_channels, out_channels)]
        for _ in range(max(0, residual_depth - 1)):
            blocks.append(ResidualUnit(out_channels, out_channels))
        blocks.append(nn.MaxPool2d(2))
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.attn1(self.stage1(x))
        x = self.attn2(self.stage2(x))
        x = self.attn3(self.stage3(x))
        x = self.attn4(self.stage4(x))
        x = self.pool(x)
        return self.classifier(x)
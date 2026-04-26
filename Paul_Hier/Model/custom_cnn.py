import warnings

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(
        self,
        n_classes=5,
        dropout=0.4,
        pretrained=False,
        base_channels=64,
        adaptive_pool_size=8,
        classifier_hidden_dim=1024,
        classifier_bottleneck_dim=256,
    ):
        super().__init__()

        if pretrained:
            warnings.warn(
                "CustomCNN does not support pretrained weights. Training from scratch.",
                RuntimeWarning,
                stacklevel=2,
            )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        def conv_bn_relu(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn_relu(3, c1),
            nn.MaxPool2d(2),
            conv_bn_relu(c1, c2),
            nn.MaxPool2d(2),
            conv_bn_relu(c2, c3),
            conv_bn_relu(c3, c3),
            nn.MaxPool2d(2),
            conv_bn_relu(c3, c4),
            conv_bn_relu(c4, c4),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size)),
        )

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
        x = self.features(x)
        return self.classifier(x)
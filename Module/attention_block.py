import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_unit import ResidualUnit

class AttentionBlock(nn.Module):
    def __init__(self, in_channel, skip=2, p=1, t=2, r=1):
        super().__init__()
        self.p = p
        self.t = t
        self.r = r
        self.skip = skip

        # Pre-activation residual units
        self.pre = nn.Sequential(*[ResidualUnit(in_channel, in_channel) for _ in range(p)])

        # Trunk branch
        self.trunk = nn.Sequential(*[ResidualUnit(in_channel, in_channel) for _ in range(t)])

        # Soft mask branch — down path
        self.down1 = nn.MaxPool2d(kernel_size=2, padding=0)
        self.down1_res = nn.Sequential(*[ResidualUnit(in_channel, in_channel) for _ in range(r)])

        # Skip connections + extra downsampling (only if skip > 1)
        self.skip_res = nn.ModuleList([
            ResidualUnit(in_channel, in_channel) for _ in range(skip - 1)
        ])
        self.down_extra = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, padding=0) for _ in range(skip - 1)
        ])
        self.down_extra_res = nn.ModuleList([
            nn.Sequential(*[ResidualUnit(in_channel, in_channel) for _ in range(r)])
            for _ in range(skip - 1)
        ])

        # Up path
        self.up_extra_res = nn.ModuleList([
            nn.Sequential(*[ResidualUnit(in_channel, in_channel) for _ in range(r)])
            for _ in range(skip - 1)
        ])
        self.up_extra = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(skip - 1)
        ])

        self.up1_res = nn.Sequential(*[ResidualUnit(in_channel, in_channel) for _ in range(r)])
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Output conv for soft mask
        self.mask_conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.mask_conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()

        # Post residual units
        self.post = nn.Sequential(*[ResidualUnit(in_channel, in_channel) for _ in range(p)])

    def forward(self, x):
        x = self.pre(x)

        # Trunk
        trunk = self.trunk(x)

        # Soft mask — down
        m = self.down1(x)
        m = self.down1_res(m)

        skip_connections = []
        if x.shape[2] % 4 == 0:
            for i in range(self.skip - 1):
                skip_connections.append(self.skip_res[i](m))
                m = self.down_extra[i](m)
                m = self.down_extra_res[i](m)

            skip_connections = list(reversed(skip_connections))

            for i in range(self.skip - 1):
                m = self.up_extra_res[i](m)
                m = self.up_extra[i](m)
                m = m + skip_connections[i]

        # Soft mask — up
        m = self.up1_res(m)
        m = self.up1(m)

        # Match mask spatial size to trunk even when odd dimensions are present.
        if m.shape[2:] != trunk.shape[2:]:
            m = F.interpolate(m, size=trunk.shape[2:], mode='bilinear', align_corners=False)

        m = self.mask_conv2(self.mask_conv1(m))
        m = self.sigmoid(m)

        # Residual attention: keep trunk path and gate salient responses.
        out = trunk * (1.0 + m)
        out = self.post(out)
        return out
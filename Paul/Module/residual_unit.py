import torch
import torch.nn as nn

class ResidualUnit(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        # Main path
        self.bn1   = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                               padding=1, stride=stride, bias=False)
        
        self.bn2   = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, 
                               padding=1, bias=False)

        # Shortcut — only project if dimensions change
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        return x + shortcut
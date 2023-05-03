"""
Generator model definition.
"""

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels

        self.down1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1),  # 64,32,32
            nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 64,16,16
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128,8,8
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256,4,4
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 2000, 1),  # 2000,4,4
            nn.ReLU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(2000, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256,8,8
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128,16,16
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64,32,32
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32,64,64
            nn.BatchNorm2d(32, 0.8),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, channels, 3, 1, 1),  # 3,64,64
            nn.Sigmoid()  # because ideal generator output is in range [0,1] (a true color image)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        self.model = nn.Sequential(
            self.down1,
            self.down2,
            self.down3,
            self.down4,
            self.middle,
            self.up1,
            self.up2,
            self.up3,
            self.up4,
            self.output
        )

    def forward(self, x: torch.Tensor):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.middle(x4)
        x6 = self.up1(x5)
        x7 = self.up2(torch.cat([x3, x6], dim=1))  # skip connection
        x8 = self.up3(torch.cat([x2, x7], dim=1))  # skip connection
        x9 = self.up4(torch.cat([x1, x8], dim=1))  # skip connection
        x10 = self.output(x9)
        return x10


if __name__ == '__main__':
    from inpainting.util import print_layer_sizes

    print_layer_sizes(Generator().model, (3, 128, 128))

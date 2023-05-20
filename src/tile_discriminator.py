"""
Discriminator model definition.
"""

import torch
from torch import nn


class TileDiscriminator(nn.Module):

    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels

        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),  # 64,32,32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 64,16,16
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128,8,8
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256,4,4
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 2000, 1),  # 2000,4,4

            nn.ConvTranspose2d(2000, 256, kernel_size=4, stride=2, padding=1),  # 256,8,8
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 256,16,16
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64,64,64
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32,64,64
            nn.BatchNorm2d(32, 0.8),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),  # 1,64,64 (single channel)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, img):
        return self.model(img)


if __name__ == '__main__':
    from src.util import print_layer_sizes

    print_layer_sizes(TileDiscriminator().model, (3, 128, 128))

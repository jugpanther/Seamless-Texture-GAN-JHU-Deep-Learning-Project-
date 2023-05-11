"""
Generator model definition.
"""
import torch
import torchvision.transforms
from torch import nn
from torchvision.transforms import Resize


class MultiResGenerator(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model16 = self._build_16x16_model()
        self.model64 = self._build_64x64_model()
        self.model128 = self._build_128x128_model()

        self.upsample16 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample64 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.resize16 = Resize(size=16, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        self.resize64 = Resize(size=64, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        self.resize128 = Resize(size=128, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)

    def _encoder_block(self, in_channels, out_channels, constant_size=False):
        if constant_size:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        return [
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        ]

    def _decoder_block(self, in_channels, out_channels, constant_size=False):
        if constant_size:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        return [
            conv,
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.5),
            nn.ReLU()
        ]

    def _build_16x16_model(self):
        in_channels = self.in_channels
        out_channels = 32
        layers = []

        # first block is different
        block = self._encoder_block(in_channels, out_channels, constant_size=True)
        block.pop(1)  # first convolution does not get a batchnorm
        layers.extend(block)

        # encoder channel doubling
        for i in range(1 + 1):
            in_channels = out_channels
            out_channels *= 2
            block = self._encoder_block(in_channels, out_channels)
            layers.extend(block)

        # bottleneck
        in_channels = out_channels
        out_channels *= 2
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=2),
            # no batchnorm
            nn.ReLU()
        ]
        layers.extend(block)

        # dcoder channel halving
        for i in range(3 + 1):
            in_channels = out_channels
            out_channels //= 2
            block = self._decoder_block(in_channels, out_channels)
            layers.extend(block)

        # final layers
        in_channels = out_channels
        layers.extend([
            self._encoder_block(in_channels, out_channels=self.out_channels, constant_size=True)[0],  # conv layer only
            nn.Sigmoid()
        ])

        # ===========================================================================

        model = nn.Sequential(*layers)

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        return model

    def _build_64x64_model(self):
        in_channels = 6
        out_channels = 64
        layers = []

        # first block is different
        block = self._encoder_block(in_channels, out_channels, constant_size=True)
        block.pop(1)  # first convolution does not get a batchnorm
        layers.extend(block)
        in_channels = out_channels
        out_channels *= 2

        # encoder channel doubling
        for i in range(2 + 1):
            block = self._encoder_block(in_channels, out_channels)
            layers.extend(block)
            in_channels = out_channels
            out_channels *= 2

        # encoder constant size operations
        out_channels = in_channels
        # for i in range(0 + 1):
        #     block = self._encoder_block(in_channels, out_channels, constant_size=True)
        #     layers.extend(block)

        # dcoder channel halving
        for i in range(2 + 1):
            in_channels = out_channels
            out_channels //= 2
            block = self._decoder_block(in_channels, out_channels)
            layers.extend(block)

        # final layers
        in_channels = out_channels
        layers.extend([
            self._encoder_block(in_channels, out_channels=self.out_channels, constant_size=True)[0],  # conv layer only
            nn.Sigmoid()
        ])

        # ===========================================================================

        model = nn.Sequential(*layers)

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        return model

    def _build_128x128_model(self):

        in_channels = 6
        out_channels = 64
        layers = []

        # first block is different
        block = self._encoder_block(in_channels, out_channels, constant_size=True)
        block.pop(1)  # first convolution does not get a batchnorm
        layers.extend(block)
        in_channels = out_channels
        out_channels *= 2

        # encoder channel doubling
        for i in range(2 + 1):  # one extra vs. 64x64 model
            block = self._encoder_block(in_channels, out_channels)
            layers.extend(block)
            in_channels = out_channels
            out_channels *= 2

        # encoder constant size operations
        out_channels = in_channels
        for i in range(0 + 1):
            block = self._encoder_block(in_channels, out_channels, constant_size=True)
            layers.extend(block)

        # dcoder channel halving
        for i in range(2 + 1):
            in_channels = out_channels
            out_channels //= 2
            block = self._decoder_block(in_channels, out_channels)
            layers.extend(block)

        # final layers
        in_channels = out_channels
        layers.extend([
            self._encoder_block(in_channels, self.out_channels, constant_size=True)[0],  # conv layer only
            nn.Sigmoid()
        ])

        # ===========================================================================

        model = nn.Sequential(*layers)

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        return model

    def forward(self, x):
        x16_in = self.resize16(x)
        x16_out = self.model16(x16_in)
        x16_out = self.upsample16(x16_out)  # upscale

        x64_in = self.resize64(x)
        x64_in = torch.cat((x64_in, x16_out), dim=1)
        x64_out = self.model64(x64_in)
        x64_out = self.upsample64(x64_out)  # upscale

        x128_in = x
        x128_in = torch.cat((x128_in, x64_out), dim=1)
        x128_out = self.model128(x128_in)

        return x128_out


if __name__ == '__main__':
    from src.util import model_sanity_check, print_layer_sizes

    gen = MultiResGenerator()
    model = gen.models[0]
    print_layer_sizes(model, (3, 16, 16))
    print()
    model = gen.models[2]
    print_layer_sizes(model, (6, 64, 64))
    print()
    model = gen.models[4]
    print_layer_sizes(model, (6, 128, 128))
    print('=' * 100)
    # gen.forward(torch.rand(1, 3, 128, 128))
    model_sanity_check(gen, (3, 128, 128), (3, 128, 128))

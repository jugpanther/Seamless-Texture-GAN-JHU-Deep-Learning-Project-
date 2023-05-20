"""
Discriminator model definition.
"""

import torchvision.models as models
from torch import nn
from torchvision.models import ResNet50_Weights


class TextureDiscriminator(nn.Module):

    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # freeze all layers of the model
        for param in self.model.parameters():
            param.requires_grad = False

        # replace the last fully connected layer with a new one that outputs a single probability value
        self.model.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.model(x)

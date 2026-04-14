import torch
import torch.nn as nn
from torchvision import models

class ResNet50Model(nn.Module):
    def __init__(self):
        super(ResNet50Model, self).__init__()

        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all layers (IMPORTANT for CPU)
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final fully connected layer
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)
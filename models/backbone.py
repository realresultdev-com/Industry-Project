"""
Neural Network Backbones for Visual Representation Learning
===========================================================
Extracts feature representations from industrial images using ResNet architectures.
Removes the final classification layer to produce feature vector h in R^D:
- ResNet-18: D = 512
- ResNet-50: D = 2048
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights
)


class BackboneEncoder(nn.Module):
    """
    Standard CNN backbone that maps image x in R^{C x H x W} to representation h in R^D.
    """
    def __init__(self, name: str = "resnet18", pretrained: bool = False):
        super().__init__()
        self.name = name.lower()
        
        if self.name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            self.feature_dim = 512
        elif self.name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {name}. Choose 'resnet18' or 'resnet50'.")

        # Extract all convolutional layers and average pooling
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional feature extractor.
        Returns:
            h: torch.Tensor of shape (batch_size, feature_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        return h

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns pre-pooling convolutional feature maps from layer4 (useful for Grad-CAM).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)
        return feat


def get_backbone(name: str = "resnet18", pretrained: bool = False):
    """
    Factory function returning the backbone encoder and feature dimension.
    """
    encoder = BackboneEncoder(name=name, pretrained=pretrained)
    return encoder, encoder.feature_dim


if __name__ == "__main__":
    enc, dim = get_backbone("resnet18")
    sample_input = torch.randn(2, 3, 200, 200)
    rep = enc(sample_input)
    print(f"[+] Backbone 'resnet18' initialized. Feature dimension: {dim}")
    print(f"[+] Output shape for input (2, 3, 200, 200): {rep.shape}")


"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
==============================================================================
Reference: Chen et al., ICML 2020.
Key components:
1. Base encoder f(.): ResNet-18 (extracts representation vector h in R^512).
2. Projection head g(.): 2-layer MLP (projects h into latent space z in R^128).
3. NT-Xent Loss: Normalized Temperature-scaled Cross-Entropy Loss over positive pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import get_backbone


class ProjectionHead(nn.Module):
    """
    Non-linear projection head g(.) mapping representation h to latent space z.
    Structure: Linear -> BatchNorm -> ReLU -> Linear.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR architecture: Backbone Encoder + Projection Head.
    """
    def __init__(self, backbone_name: str = "resnet18", proj_dim: int = 128):
        super().__init__()
        self.encoder, self.feature_dim = get_backbone(backbone_name, pretrained=False)
        self.projector = ProjectionHead(in_dim=self.feature_dim, hidden_dim=self.feature_dim, out_dim=proj_dim)

    def forward(self, x: torch.Tensor):
        """
        Extracts representation h and projection z.
        """
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=-1)
        return h, z


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (InfoNCE).
    Operates on a batch of positive pairs (z_i, z_j) where each image has two augmented views.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Tensor of shape (B, D) from first augmented view
            z_j: Tensor of shape (B, D) from second augmented view
        Returns:
            scalar contrastive loss
        """
        batch_size = z_i.shape[0]
        
        # Ensure unit L2 normalization
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Concatenate both views: shape (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute cosine similarity matrix: shape (2B, 2B)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create self-similarity mask to eliminate diagonal (z_k with itself)
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e9)

        # Target positive labels:
        # For item i in [0..B-1], positive is i + B
        # For item i in [B..2B-1], positive is i - B
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)

        # Cross-entropy loss across all negative and positive pairs
        loss = F.cross_entropy(sim_matrix, targets)
        return loss


if __name__ == "__main__":
    model = SimCLR(backbone_name="resnet18")
    loss_fn = NTXentLoss(temperature=0.5)
    
    x1 = torch.randn(4, 3, 200, 200)
    x2 = torch.randn(4, 3, 200, 200)
    
    _, z1 = model(x1)
    _, z2 = model(x2)
    
    loss = loss_fn(z1, z2)
    print(f"[+] SimCLR forward pass and NT-Xent loss calculated successfully: {loss.item():.4f}")


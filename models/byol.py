"""
BYOL: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning
============================================================================
Reference: Grill et al., NeurIPS 2020.
Key components:
1. Online Network: Backbone f_theta, Projector g_theta, Predictor q_theta.
2. Target Network: Backbone f_xi, Projector g_xi (updated via Exponential Moving Average).
3. Non-contrastive normalized Mean Squared Error loss with stop-gradient on target network.
Does not require negative pairs or large batch sizes.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import get_backbone


class MLP(nn.Module):
    """
    Multi-layer Perceptron used for Projector and Predictor in BYOL.
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


class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent (BYOL) Architecture.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        proj_dim: int = 128,
        hidden_dim: int = 512,
        ema_decay: float = 0.99
    ):
        super().__init__()
        self.ema_decay = ema_decay
        
        # 1. Online Network (Backbone + Projector + Predictor)
        self.online_encoder, self.feature_dim = get_backbone(backbone_name, pretrained=False)
        self.online_projector = MLP(self.feature_dim, hidden_dim, proj_dim)
        self.predictor = MLP(proj_dim, hidden_dim, proj_dim)

        # 2. Target Network (Encoder + Projector, no predictor)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target network gradients (it is updated only via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        """
        Exponential Moving Average (EMA) update:
        xi <- tau * xi + (1 - tau) * theta
        """
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data = self.ema_decay * target_p.data + (1.0 - self.ema_decay) * online_p.data

        for online_p, target_p in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_p.data = self.ema_decay * target_p.data + (1.0 - self.ema_decay) * online_p.data

    @staticmethod
    def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes Mean Squared Error between normalized predictions and target representations:
        L(x, y) = 2 - 2 * <x, y> / (||x||_2 * ||y||_2)
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2.0 - 2.0 * (x * y).sum(dim=-1).mean()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Symmetric forward pass:
        Loss = 0.5 * ( L(pred(online(x1)), target(x2)) + L(pred(online(x2)), target(x1)) )
        """
        # Online path for view 1
        h1 = self.online_encoder(x1)
        z1 = self.online_projector(h1)
        p1 = self.predictor(z1)

        # Online path for view 2
        h2 = self.online_encoder(x2)
        z2 = self.online_projector(h2)
        p2 = self.predictor(z2)

        # Target path (with stop gradient)
        with torch.no_grad():
            t_h1 = self.target_encoder(x1)
            t_z1 = self.target_projector(t_h1)

            t_h2 = self.target_encoder(x2)
            t_z2 = self.target_projector(t_h2)

        # Symmetric regression loss
        loss1 = self.regression_loss(p1, t_z2.detach())
        loss2 = self.regression_loss(p2, t_z1.detach())
        loss = 0.5 * (loss1 + loss2)

        return loss, h1


if __name__ == "__main__":
    model = BYOL(backbone_name="resnet18")
    v1 = torch.randn(4, 3, 200, 200)
    v2 = torch.randn(4, 3, 200, 200)
    
    loss, rep = model(v1, v2)
    model.update_target_network()
    print(f"[+] BYOL forward pass, EMA update, and loss computed: {loss.item():.4f}")


"""
baseline_models.py (placeholder)

Add simple CNN/RNN/Transformer baselines for comparison.
"""

import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, d_model: int = 128, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)

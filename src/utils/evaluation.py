"""
evaluation.py (placeholder)

Metrics and confusion matrix helpers.
"""

from typing import Dict
import torch

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == targets).float().mean().item())

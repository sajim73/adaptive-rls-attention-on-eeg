"""
train_rls.py (work-in-progress)

Contains training/evaluation loops for RLS-based models.
Replace placeholders with logic extracted from your RLS notebooks.
"""

from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch_rls(model, loader: DataLoader, optimizer, device) -> Dict[str, float]:
    model.train()
    loss_sum, n = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss()
    for batch in tqdm(loader, desc="Train (RLS)", leave=False):
        x, y = batch["x"].to(device), batch["y"].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
    return {"loss": loss_sum / max(n, 1)}

@torch.no_grad()
def evaluate_rls(model, loader: DataLoader, device) -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(loader, desc="Eval (RLS)", leave=False):
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    acc = correct / max(total, 1)
    return {"accuracy": acc}

def subject_dependent_experiment_rls():
    # Placeholder hook for subject-dependent protocol
    # Wire up Dataset -> DataLoader -> train/eval here.
    pass

def cross_subject_experiment_rls():
    # Placeholder hook for cross-subject protocol
    pass

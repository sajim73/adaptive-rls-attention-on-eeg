"""
data_loader.py (placeholder)

Provide a SEED-VII compatible Dataset returning dicts:
{
  "x": Tensor (B, T, D),
  "y": Tensor (B,)
}
Replace with your concrete loader and preprocessing pipeline.
"""

from torch.utils.data import Dataset

class SEEDVII_Dataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root
        self.split = split
        self._size = 100  # placeholder

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        # Return dummy sample; replace with real loading
        import torch
        x = torch.randn(128, 128)  # (T, D) placeholder
        y = torch.randint(0, 7, (1,)).item()
        return {"x": x, "y": y}

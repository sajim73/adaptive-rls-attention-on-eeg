import torch
from src.models.rls_attention import Phase1RLSModel

def test_forward_shapes():
    model = Phase1RLSModel(d_model=128, num_classes=7)
    x = torch.randn(2, 128, 128)  # (B, T, D)
    logits = model(x)
    assert logits.shape == (2, 7)

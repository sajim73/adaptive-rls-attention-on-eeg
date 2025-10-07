"""
Baseline models for comparison with RLS Attention models.
Contains standard implementations for benchmarking performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class SimpleFFN(nn.Module):
    """Simple Feed-Forward Network baseline."""

    def __init__(self, input_dim: int = 310, hidden_dim: int = 256, 
                 num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.name = "SimpleFFN"

    def forward(self, x):
        # Handle sequence input by averaging
        if x.dim() == 3:
            x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


class StandardTransformer(nn.Module):
    """Standard Transformer baseline."""

    def __init__(self, input_dim: int = 310, d_model: int = 256, 
                 num_heads: int = 8, num_layers: int = 3, num_classes: int = 7):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, num_classes)
        self.name = "StandardTransformer"

    def forward(self, x):
        batch_size = x.size(0)

        # Handle different input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Project to model dimension
        x = self.input_projection(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        cls_output = x[:, 0]

        return self.classifier(cls_output)


class LinearRegression(nn.Module):
    """Simple linear regression baseline."""

    def __init__(self, input_dim: int = 310, num_classes: int = 7):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.name = "LinearRegression"

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.linear(x)


class CNN1D(nn.Module):
    """1D CNN baseline for EEG sequence data."""

    def __init__(self, input_dim: int = 310, num_classes: int = 7, 
                 hidden_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_classes)
        )
        self.name = "CNN1D"

    def forward(self, x):
        # Handle different input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension for conv1d
        elif x.dim() == 3:
            x = x.mean(dim=1).unsqueeze(1)  # Average over sequence, add channel

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # Global average pooling
        return self.classifier(x)


class LSTM_Classifier(nn.Module):
    """LSTM baseline classifier."""

    def __init__(self, input_dim: int = 310, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.name = "LSTM"

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        return self.classifier(h_n[-1])


class FixedRLSLinearModel(nn.Module):
    """
    RLS-enhanced linear model with NO in-place operations during forward pass.
    This is a fixed version that addresses autograd issues.
    """

    def __init__(self, n_features: int = 310, n_classes: int = 7, 
                 lambda_reg: float = 0.01, forgetting_factor: float = 0.99, 
                 use_rls: bool = True):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.lambda_reg = lambda_reg
        self.forgetting_factor = forgetting_factor
        self.use_rls = use_rls

        # Standard linear layer (baseline performance)
        self.linear = nn.Linear(n_features, n_classes)

        # RLS parameters for feature weighting
        if use_rls:
            # Feature importance weights learned via RLS
            self.register_buffer('feature_weights', torch.ones(n_features))

            # RLS matrices (simplified for feature weighting)
            self.register_buffer('P', torch.eye(n_features) / lambda_reg)
            self.register_buffer('rls_count', torch.tensor(0.0))

        # Input normalization
        self.input_norm = nn.LayerNorm(n_features)

        # Increased regularization to prevent overfitting
        self.dropout = nn.Dropout(0.5)  # Increased from 0.1 to 0.5

        self.name = "FixedRLSLinear"

    def forward(self, x, labels=None):
        """FIXED: No in-place operations during forward pass."""
        # Handle sequence input by averaging (like successful baselines)
        if x.dim() == 3:
            x = x.mean(dim=1)  # (batch, seq, features) -> (batch, features)

        # Normalize input
        x = self.input_norm(x)

        # Apply learned feature weighting (RLS component) - OUT-OF-PLACE
        if self.use_rls:
            x = x * self.feature_weights.unsqueeze(0)  # Create new tensor, don't modify x

        # Apply dropout
        x = self.dropout(x)

        # Linear classification (matching successful baseline)
        logits = self.linear(x)

        # Return logits and features for RLS update (no in-place updates here!)
        return logits, x.detach() if self.use_rls else None

    def rls_update(self, features, labels, logits):
        """FIXED: RLS update called AFTER backward pass under torch.no_grad()."""
        if not self.use_rls or not self.training:
            return

        with torch.no_grad():
            # Calculate prediction error for RLS update
            pred_probs = F.softmax(logits, dim=1)
            true_probs = F.one_hot(labels, num_classes=self.n_classes).float()
            error = pred_probs - true_probs  # (batch_size, n_classes)

            # Simplified RLS update using batch statistics
            batch_size = features.size(0)

            # Use batch mean for stability
            x_mean = features.mean(dim=0)  # (n_features,)
            error_mean = error.mean().item()  # Scalar error

            # RLS update equations
            Px = self.P @ x_mean.unsqueeze(-1)  # (n_features, 1)
            denominator = self.forgetting_factor + x_mean @ Px.squeeze()
            K = Px / (denominator + 1e-8)  # Kalman gain

            # Update feature weights - SAFE buffer updates
            weight_update = K.squeeze() * error_mean * 0.01  # Small learning rate
            new_weights = self.feature_weights + weight_update
            self.feature_weights.copy_(torch.clamp(new_weights, min=0.01))

            # Update P matrix - SAFE buffer update  
            outer_product = torch.outer(K.squeeze(), x_mean)
            new_P = (self.P - outer_product) / self.forgetting_factor
            self.P.copy_(new_P)

            self.rls_count += 1


class FixedRLSDeepModel(nn.Module):
    """
    Deeper RLS model with heavy regularization to prevent overfitting.
    """

    def __init__(self, n_features: int = 310, n_classes: int = 7, 
                 hidden_dim: int = 128, dropout_rate: float = 0.6):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        # Input processing
        self.input_norm = nn.LayerNorm(n_features)

        # Feature selection/weighting (RLS-inspired)
        self.feature_selector = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid()  # Soft feature selection
        )

        # Main classifier with HEAVY regularization
        self.classifier = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),  # FIXED: No inplace operations
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=False),  # FIXED: No inplace operations
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, n_classes)
        )

        self.name = "FixedRLSDeep"

    def forward(self, x, labels=None):
        # Handle sequence input
        if x.dim() == 3:
            x = x.mean(dim=1)

        # Normalize
        x = self.input_norm(x)

        # Feature selection (RLS-inspired adaptive weighting) - OUT-OF-PLACE
        feature_weights = self.feature_selector(x)
        x = x * feature_weights  # Creates new tensor

        # Classification
        logits = self.classifier(x)

        return logits, feature_weights.detach()


def train_baseline_model(model, train_loader, val_loader, epochs: int = 30, 
                        lr: float = 1e-3, device: str = 'cuda') -> Tuple[float, List, List]:
    """
    Train any baseline model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        Tuple of (best_val_acc, train_losses, val_accs)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    train_losses, val_accs = [], []

    print(f"Training {model.name}...")

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(eeg)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                labels = batch['label'].to(device)
                outputs = model(eeg)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / total
        train_losses.append(total_loss / len(train_loader))
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {train_losses[-1]:.4f}, Val Acc {val_acc:.2f}%")

    print(f"{model.name} Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, train_losses, val_accs


def train_fixed_rls_model(model, train_loader, val_loader, epochs: int = 50, 
                         lr: float = 1e-4, weight_decay: float = 0.5, 
                         device: str = 'cuda', patience: int = 10) -> Tuple[float, List, List]:
    """
    FIXED: Training function with proper RLS updates after backward pass.

    Args:
        model: RLS model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        patience: Early stopping patience

    Returns:
        Tuple of (best_val_acc, train_losses, val_accs)
    """
    model = model.to(device)

    # FIXED: Stronger regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_accs = [], []

    print(f"Training {model.name}...")

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # FIXED: Forward pass returns features for RLS update
            outputs, features = model(eeg, labels)

            loss = criterion(outputs, labels)

            # L1 regularization on feature weights if available
            if features is not None and hasattr(model, 'feature_weights'):
                l1_reg = torch.mean(torch.abs(model.feature_weights))
                loss += 0.001 * l1_reg

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # FIXED: RLS update AFTER optimizer.step() under no_grad
            if hasattr(model, 'rls_update') and features is not None:
                model.rls_update(features, labels, outputs.detach())

            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                labels = batch['label'].to(device)

                outputs, _ = model(eeg)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_acc)

        # FIXED: Early stopping at reasonable performance to prevent overfitting
        if val_acc > 70:  # Stop at 70% to prevent overfitting
            print(f"Early stopping at {val_acc:.2f}% to prevent overfitting")
            patience_counter = patience  # Trigger early stopping

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # FIXED: Safe model saving
            torch.save(model.state_dict(), f'best_{model.name}.pth', 
                      use_new_zipfile_serialization=False)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}, Val Acc {val_acc:.2f}%, "
                  f"LR {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # FIXED: Safe model loading
    try:
        model.load_state_dict(torch.load(f'best_{model.name}.pth', weights_only=True))
    except:
        model.load_state_dict(torch.load(f'best_{model.name}.pth'))

    print(f"{model.name} Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, train_losses, val_accs


def run_baseline_comparison(data_dir: str = '.', subset_ratio: float = 0.5) -> Dict[str, Dict]:
    """
    Run comprehensive baseline comparison.

    Args:
        data_dir: Data directory path
        subset_ratio: Fraction of data to use

    Returns:
        Dictionary of results for each model
    """
    print("=" * 80)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("=" * 80)

    # This would need the actual dataset loading code
    # For now, we'll return a template
    results = {
        'LinearRegression': {
            'val_acc': 0.0,
            'test_acc': 0.0,
            'test_f1': 0.0
        },
        'SimpleFFN_128': {
            'val_acc': 0.0,
            'test_acc': 0.0,
            'test_f1': 0.0
        },
        'SimpleFFN_256': {
            'val_acc': 0.0,
            'test_acc': 0.0,
            'test_f1': 0.0
        },
        'StandardTransformer': {
            'val_acc': 0.0,
            'test_acc': 0.0,
            'test_f1': 0.0
        }
    }

    # Summary comparison
    print("=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Val Acc':<10} {'Test Acc':<10} {'Test F1':<10}")
    print("-" * 60)

    for name, metrics in results.items():
        print(f"{name:<20} {metrics['val_acc']:<10.2f} {metrics['test_acc']:<10.2f} "
              f"{metrics['test_f1']:<10.2f}")

    # Random baseline
    random_acc = 100.0 / 7  # 7 classes for SEED-VII
    print(f"{'Random Baseline':<20} {random_acc:<10.2f} {random_acc:<10.2f} "
          f"{random_acc:<10.2f}")

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("1. Compare your RLS model against these baselines")
    print("2. If RLS performs worse than simple baselines, check implementation")
    print("3. Expected performance ranges:")
    print("   - Random baseline: ~14.3% (1/7 classes)")
    print("   - Simple baselines: 30-50%")
    print("   - Advanced models: 50-70%")

    return results


# Factory function to create baseline models
def create_baseline_model(model_name: str, input_dim: int = 310, 
                         num_classes: int = 7, **kwargs) -> nn.Module:
    """
    Factory function to create baseline models.

    Args:
        model_name: Name of the model to create
        input_dim: Input feature dimension
        num_classes: Number of output classes
        **kwargs: Additional model-specific parameters

    Returns:
        Instantiated model
    """
    models = {
        'linear': LinearRegression,
        'ffn': SimpleFFN,
        'transformer': StandardTransformer,
        'cnn1d': CNN1D,
        'lstm': LSTM_Classifier,
        'rls_linear': FixedRLSLinearModel,
        'rls_deep': FixedRLSDeepModel
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    model_class = models[model_name.lower()]
    return model_class(input_dim=input_dim, num_classes=num_classes, **kwargs)

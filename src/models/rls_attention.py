import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalRLSAttention(nn.Module):
    def __init__(self, d_model, lambda_reg=0.01, forgetting_factor=0.99, eps=1e-8):
        super(TemporalRLSAttention, self).__init__()
        self.d_model = d_model
        self.lambda_reg = lambda_reg
        self.forgetting_factor = forgetting_factor
        self.eps = eps

        P0 = torch.eye(d_model) / lambda_reg
        self.register_buffer('P', P0)
        self.register_buffer('theta', torch.zeros(d_model, 1))

    def forward(self, x: torch.Tensor, target: torch.Tensor = None):
        """
        x: (batch_size, seq_len, d_model) or (batch_size, d_model)
        target: optional (batch_size, seq_len) or (batch_size)
        """
        # Normalize input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,D)
            if target is not None and target.dim() == 1:
                target = target.unsqueeze(1)

        B, T, D = x.shape
        weights = []

        for t in range(T):
            x_t = x[:, t, :].unsqueeze(-1)  # (B,D,1)
            P_exp = self.P.unsqueeze(0).expand(B, D, D)
            Px = torch.bmm(P_exp, x_t)      # (B,D,1)
            denom = self.forgetting_factor + torch.bmm(x_t.transpose(-2, -1), Px).squeeze(-1)
            K = Px / (denom.unsqueeze(-1) + self.eps)  # (B,D,1)

            y_pred = torch.bmm(
                x_t.transpose(-2, -1),
                self.theta.unsqueeze(0).expand(B, D, 1)
            ).squeeze(-1)  # (B,)

            if target is not None:
                err = target[:, t].unsqueeze(-1) - y_pred.unsqueeze(-1)
                delta = torch.mean(K * err.unsqueeze(-1), dim=0)
                self.theta = self.theta + delta
                KAt = torch.bmm(K, x_t.transpose(-2, -1))
                self.P = (self.P - torch.mean(KAt, dim=0)) / self.forgetting_factor

            weights.append(y_pred)

        attn = torch.stack(weights, dim=1)           # (B,T)
        attn_norm = F.softmax(attn, dim=1)           # normalize for weighting
        output = torch.sum(x * attn_norm.unsqueeze(-1), dim=1)  # (B,D)
        return output, attn_norm


class Phase1RLSModel(nn.Module):
    def __init__(self, n_channels=310, d_model=256, n_classes=7,
                 lambda_reg=0.01, forgetting_factor=0.99):
        super(Phase1RLSModel, self).__init__()
        self.embed = nn.Linear(n_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.rls = TemporalRLSAttention(d_model, lambda_reg, forgetting_factor)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, eeg, target=None):
        # Handle 2D input
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(1)  # (B,1,D)
        x = self.embed(eeg)        # (B,T,d_model)
        x = self.norm(x)
        features, attn = self.rls(x, target)
        logits = self.head(features)
        return logits, attn

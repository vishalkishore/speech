import torch
import torch.nn as nn

class CausalFreqNorm(nn.Module):
    """
    Normalizes over frequency axis only (dim=-1 of [B, C, T, F]).
    Per (B, C, t) → mean/var computed over F only.
    Zero temporal mixing → fully causal, identical at train & streaming inference.
    """
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.weight = self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        mean = x.mean(dim=-1, keepdim=True)               # [B, C, T, 1]
        var  = x.var(dim=-1, keepdim=True, unbiased=False) # [B, C, T, 1]
        x = (x - mean) / (var + self.eps).sqrt()
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


import os


class MagNorm(nn.Module):
    """
    Causal per-bin EMA normalization for log-magnitude input [B, T, F].
    stateful=True  → streaming inference (state carried across frames within utterance).
    stateful=False → training (resets each forward call).
    """
    def __init__(self, alpha_attack: float = 0.95, alpha_decay: float = 0.995, eps: float = 1.0,
                 stateful: bool = False, dynamic_var: bool = True):
        super().__init__()
        self.alpha_attack = alpha_attack
        self.alpha_decay = alpha_decay
        self.eps = eps
        self.stateful = stateful
        self.dynamic_var = dynamic_var
        self.mu  = None
        self.var = None

        # Register as buffers so they move with .to(device) automatically
        self.register_buffer('dataset_mu',  None)
        self.register_buffer('dataset_var', None)

        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'mag_stats.pt'
        )
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu')
            self.dataset_mu  = stats['mean']
            self.dataset_var = stats['std'].pow(2)
            print(f"MagNorm: Loaded dataset stats from {stats_path}")

    def _get_init(self, B, F, device):
        if self.dataset_mu is not None:
            # Start EMA at actual dataset statistics — no warm-up needed
            mu  = self.dataset_mu.to(device)[None].expand(B, F)
            var = self.dataset_var.to(device)[None].expand(B, F)
            return mu.clone(), var.clone()

        # Fallback heuristic: -60 → -90 dB across freq bins
        step = (-90.0 - (-60.0)) / (F - 1)
        mu   = (-60.0 + torch.arange(F, device=device) * step)[None].expand(B, F)
        var  = torch.full_like(mu, 40 ** 2)
        return mu.clone(), var.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]  (log-magnitude in dB)
        B, T, F = x.shape

        if self.mu is None or not self.stateful:
            mu, var = self._get_init(B, F, x.device)
            # Phantom warm-up only needed when dataset stats are unavailable
            if self.dataset_mu is None:
                phantom_frames = (
                    torch.randint(0, 500, (1,)).item() if self.training else 250
                )
                phantom_signal = torch.randn(B, F, device=x.device) * 15.0 - 30.0
                for _ in range(phantom_frames):
                    alpha_t = torch.where(phantom_signal > mu,
                                          torch.tensor(self.alpha_attack, device=x.device),
                                          torch.tensor(self.alpha_decay, device=x.device))
                    mu = alpha_t * mu + (1 - alpha_t) * phantom_signal
                    if self.dynamic_var:
                        var = alpha_t * var + (1 - alpha_t) * (phantom_signal - mu) ** 2
        else:
            mu, var = self.mu, self.var

        x_norm = []
        for t in range(T):
            alpha_t = torch.where(x[:, t] > mu,
                                  torch.tensor(self.alpha_attack, device=x.device),
                                  torch.tensor(self.alpha_decay, device=x.device))
            mu = alpha_t * mu + (1 - alpha_t) * x[:, t]
            if self.dynamic_var:
                var = alpha_t * var + (1 - alpha_t) * (x[:, t] - mu) ** 2
            x_norm.append((x[:, t] - mu) / (var + self.eps).sqrt())
        x_norm = torch.stack(x_norm, dim=1)  # [B, T, F]

        self.mu  = mu.detach()  if self.stateful else None
        self.var = var.detach() if self.stateful else None
        return x_norm

    def reset_state(self):
        self.mu = self.var = None
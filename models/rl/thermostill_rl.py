"""Policy-side neural modules used by ThermoStill.

The public path uses a decomposition-based temporal encoder and
Beta-distributed teacher weights, matching the method description in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """Split an input sequence into trend and seasonal components."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original = x
        pad = (self.kernel_size - 1) // 2
        x = x.transpose(1, 2)
        x = F.pad(x, (pad, pad), mode="replicate")
        trend = self.avg_pool(x).transpose(1, 2)
        seasonal = original - trend
        return trend, seasonal


class DecompositionEncoder(nn.Module):
    """Encode the historical context using the trend/seasonal decomposition path."""

    def __init__(self, context_length: int, n_features: int, hidden_dim: int, kernel_size: int):
        super().__init__()
        input_dim = context_length * n_features
        self.decomposition = SeriesDecomposition(kernel_size=kernel_size)
        self.trend_proj = nn.Linear(input_dim, hidden_dim)
        self.seasonal_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend, seasonal = self.decomposition(x)
        return self.trend_proj(trend.reshape(trend.shape[0], -1)) + self.seasonal_proj(
            seasonal.reshape(seasonal.shape[0], -1)
        )


class ThermoStillActor(nn.Module):
    """Actor network that parameterizes one Beta distribution per teacher."""

    def __init__(self, args):
        super().__init__()
        self.n_teachers = len(args.tsfm_name_list)
        self.encoder = DecompositionEncoder(
            context_length=args.context_length,
            n_features=args.n_features,
            hidden_dim=args.input_enc_dim,
            kernel_size=args.trend_kernel_size,
        )
        self.fc1 = nn.Linear(args.input_enc_dim + 2 * self.n_teachers, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 2 * self.n_teachers)

    def forward(self, batch_x: torch.Tensor, teacher_gt_error: torch.Tensor, teacher_student_gap: torch.Tensor):
        sample_embed = self.encoder(batch_x)
        state = torch.cat([sample_embed, teacher_gt_error, teacher_student_gap], dim=-1)
        hidden = F.relu(self.fc1(state))
        stats = self.fc2(hidden)
        alpha_raw, beta_raw = torch.chunk(stats, 2, dim=-1)
        return F.softplus(alpha_raw) + 1e-3, F.softplus(beta_raw) + 1e-3


class ThermoStillCritic(nn.Module):
    """Value network used by PPO to estimate state value."""

    def __init__(self, args):
        super().__init__()
        self.n_teachers = len(args.tsfm_name_list)
        self.encoder = DecompositionEncoder(
            context_length=args.context_length,
            n_features=args.n_features,
            hidden_dim=args.input_enc_dim,
            kernel_size=args.trend_kernel_size,
        )
        self.fc1 = nn.Linear(args.input_enc_dim + 2 * self.n_teachers, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch_x: torch.Tensor, teacher_gt_error: torch.Tensor, teacher_student_gap: torch.Tensor):
        sample_embed = self.encoder(batch_x)
        state = torch.cat([sample_embed, teacher_gt_error, teacher_student_gap], dim=-1)
        hidden = F.relu(self.fc1(state))
        return self.fc2(hidden).squeeze(-1)

"""Shared utilities for RC student models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _safe_fraction(init: float, lower: float, upper: float) -> float:
    if not lower < init < upper:
        raise ValueError(f"Initial value {init} must lie inside bounds ({lower}, {upper}).")
    fraction = (init - lower) / (upper - lower)
    return min(max(fraction, 1e-6), 1.0 - 1e-6)


def make_bounded_parameter(init: float, lower: float, upper: float) -> nn.Parameter:
    """Create a raw parameter that is mapped into a fixed physical interval."""
    fraction = _safe_fraction(init, lower, upper)
    raw_value = math.log(fraction / (1.0 - fraction))
    return nn.Parameter(torch.tensor(raw_value, dtype=torch.float32))


def bounded_value(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Map an unconstrained raw parameter to a bounded physical interval."""
    return lower + (upper - lower) * torch.sigmoid(raw)

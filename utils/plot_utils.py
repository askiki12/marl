"""Plot helpers for experiment curves."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def plot_learning_curves(metrics: Mapping[str, Sequence[float]], output_path: str | Path) -> None:
    """Placeholder plotting hook.

    TODO: add matplotlib-based curve rendering once training logs are ready.
    """

    _ = metrics
    _ = output_path
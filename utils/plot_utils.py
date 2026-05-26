"""Plot helpers for experiment curves."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def _format_metric_label(metric_name: str) -> str:
    parts = metric_name.replace("_", " ").strip().split()
    return " ".join(part.capitalize() for part in parts) if parts else metric_name


def plot_learning_curves(
    metrics: Mapping[str, Sequence[float]],
    output_path: str | Path,
    *,
    title: str = "Learning Curves",
    xlabel: str = "Episode",
    ylabel: str | None = None,
) -> None:
    """Render and save learning curves.

    The function accepts a mapping from curve name to a sequence of scalars
    and writes a PNG figure to ``output_path``.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or "Value")
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric_name, values in metrics.items():
        if not values:
            continue
        ax.plot(range(1, len(values) + 1), values, label=metric_name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif len(metrics) == 1:
        ax.set_ylabel(_format_metric_label(next(iter(metrics.keys()))))
    else:
        ax.set_ylabel("Metric value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_directories(root: Path | None = None) -> dict[str, Path]:
    root = root or project_root()
    paths = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "metrics": root / "data" / "processed" / "metrics",
        "figures_before": root / "reports" / "figures" / "before_process",
        "figures_after": root / "reports" / "figures" / "after_process",
        "figures_clustering": root / "reports" / "figures" / "clustering",
        "models": root / "models",
        "reports": root / "reports",
        "config": root / "config",
        "log": root / "log",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    with (root / "config" / "clustering_config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def find_raw_csv(root: Path | None = None, pattern: str = "data/raw/*.csv") -> Path:
    root = root or project_root()
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV files found with pattern {pattern!r}")
    return matches[0]


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_figure(fig: plt.Figure, path: Path, dpi: int = 160) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def status(message: str) -> None:
    print(f"[lab03] {message}")


def write_training_log(
    root: Path,
    config: dict[str, Any] | None = None,
    n_samples: int | None = None,
    selected_algorithm: str | None = None,
    selected_feature_set: str | None = None,
    selected_scaler: str | None = None,
    selected_params: str | None = None,
    n_clusters: int | None = None,
    silhouette: float | None = None,
    n_models_evaluated: int | None = None,
    cluster_distribution: dict[int, int] | None = None,
    extra_lines: list[str] | None = None,
) -> Path:
    log_dir = root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"train_{ts}.log"
    lines = [
        "=== Training Log ===",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Project: {root.name}",
        "",
    ]
    if config:
        lines.append("--- Config ---")
        lines.append(f"Feature sets: {config.get('feature_sets')}")
        lines.append(f"Scalers: {config.get('scalers')}")
        lines.append(f"K-Means K range: {config.get('kmeans_k_range')}")
        lines.append(f"GMM K range: {config.get('gmm_k_range')}")
        lines.append("")
    if n_samples is not None:
        lines.append(f"Total samples: {n_samples}")
        lines.append("")
    if selected_algorithm:
        lines.append("--- Selected Model ---")
        lines.append(f"Algorithm: {selected_algorithm}")
        if selected_feature_set:
            lines.append(f"Feature set: {selected_feature_set}")
        if selected_scaler:
            lines.append(f"Scaler: {selected_scaler}")
        if selected_params:
            lines.append(f"Parameters: {selected_params}")
        if n_clusters:
            lines.append(f"Number of clusters: {n_clusters}")
        if silhouette:
            lines.append(f"Silhouette: {silhouette}")
        lines.append("")
    if n_models_evaluated is not None:
        lines.append(f"Models evaluated: {n_models_evaluated}")
        lines.append("")
    if cluster_distribution:
        lines.append("--- Cluster Distribution ---")
        total = sum(cluster_distribution.values())
        for cid, cnt in sorted(cluster_distribution.items()):
            pct = cnt / total * 100
            lines.append(f"  Cluster {cid}: {cnt} ({pct:.1f}%)")
        lines.append("")
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

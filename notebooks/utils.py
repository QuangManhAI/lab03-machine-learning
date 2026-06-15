from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

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

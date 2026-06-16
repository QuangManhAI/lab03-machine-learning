from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import save_figure


sns.set_theme(style="whitegrid", context="notebook")


def plot_missing_values(df: pd.DataFrame, output_dir: Path) -> Path:
    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing.index, y=missing.values, ax=ax, color="#4C78A8")
    ax.set_title("Missing Values By Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=35)
    return save_figure(fig, output_dir / "missing_values.png")


def plot_duplicate_summary(duplicate_checks: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=duplicate_checks, x="check", y="count", ax=ax, color="#72B7B2")
    ax.set_title("Duplicate Checks")
    ax.set_xlabel("Check")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    return save_figure(fig, output_dir / "duplicate_summary.png")


def plot_boxplots(df: pd.DataFrame, output_dir: Path, suffix: str = "before") -> Path:
    numeric = ["age", "annual_income", "spending_score"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column in zip(axes, numeric):
        sns.boxplot(y=df[column], ax=ax, color="#F58518")
        ax.set_title(column.replace("_", " ").title())
        ax.set_ylabel("")
    fig.suptitle("Numeric Feature Boxplots", y=1.04)
    return save_figure(fig, output_dir / f"boxplots_{suffix}.png")


def plot_distributions(df: pd.DataFrame, output_dir: Path, suffix: str = "before") -> list[Path]:
    paths = []
    numeric = ["age", "annual_income", "spending_score"]
    for column in numeric:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df[column], kde=True, ax=ax, color="#4C78A8", bins=30)
        ax.set_title(f"Distribution Of {column.replace('_', ' ').title()}")
        ax.set_xlabel(column.replace("_", " ").title())
        ax.set_ylabel("Customers")
        paths.append(save_figure(fig, output_dir / f"distribution_{column}_{suffix}.png"))
    return paths


def plot_gender_distribution(df: pd.DataFrame, output_dir: Path, suffix: str = "before") -> Path | None:
    if "gender" not in df:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="gender", ax=ax, hue="gender", palette=["#54A24B", "#E45756"], legend=False)
    ax.set_title("Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Customers")
    return save_figure(fig, output_dir / f"gender_distribution_{suffix}.png")


def plot_relationships(df: pd.DataFrame, output_dir: Path, suffix: str = "before") -> list[Path]:
    pairs = [
        ("annual_income", "spending_score"),
        ("age", "spending_score"),
        ("age", "annual_income"),
    ]
    paths = []
    for x_col, y_col in pairs:
        fig, ax = plt.subplots(figsize=(7, 5))
        hue = "gender" if "gender" in df else None
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue,
            alpha=0.45,
            s=22,
            linewidth=0,
            ax=ax,
            palette="Set2",
        )
        ax.set_title(f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}")
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        paths.append(save_figure(fig, output_dir / f"scatter_{x_col}_vs_{y_col}_{suffix}.png"))
    return paths


def plot_pairplot(df: pd.DataFrame, output_dir: Path, suffix: str = "before") -> Path:
    sample = df.sample(min(len(df), 2500), random_state=42)
    grid = sns.pairplot(
        sample[["age", "annual_income", "spending_score", "gender"]],
        hue="gender",
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.45, "s": 18, "linewidth": 0},
    )
    grid.fig.suptitle("Numeric Feature Scatter Matrix", y=1.02)
    path = output_dir / f"pairplot_{suffix}.png"
    grid.fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(grid.fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path, suffix: str = "before") -> Path:
    corr = df[["age", "annual_income", "spending_score"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    return save_figure(fig, output_dir / f"correlation_heatmap_{suffix}.png")

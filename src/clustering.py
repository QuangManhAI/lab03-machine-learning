from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from model_from_scratch import KMeans, HierarchicalClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils import save_figure


@dataclass
class ModelResult:
    estimator: Any
    transformer: Any
    labels: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]


def _label_stats(labels: np.ndarray) -> tuple[int, float]:
    labels = np.asarray(labels)
    non_noise = labels[labels != -1]
    n_clusters = len(set(non_noise))
    noise_rate = float(np.mean(labels == -1)) if len(labels) else 0.0
    return n_clusters, noise_rate


def compute_metrics(X: np.ndarray, labels: np.ndarray, random_seed: int = 42) -> dict[str, float | None]:
    n_clusters, noise_rate = _label_stats(labels)
    valid = labels != -1
    X_valid = X[valid]
    labels_valid = labels[valid]
    if n_clusters < 2 or len(np.unique(labels_valid)) < 2 or len(labels_valid) <= n_clusters:
        return {
            "n_clusters": n_clusters,
            "noise_rate": noise_rate,
            "silhouette": None,
        }
    sample_size = min(len(labels_valid), 5000)
    silhouette = silhouette_score(X_valid, labels_valid, sample_size=sample_size, random_state=random_seed)
    return {
        "n_clusters": n_clusters,
        "noise_rate": noise_rate,
        "silhouette": float(silhouette),
    }


def run_kmeans_grid(
    matrices: dict[tuple[str, str], tuple[np.ndarray, list[str], Any]],
    k_values: range,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[int, ModelResult]]:
    rows = []
    models: dict[int, ModelResult] = {}
    model_id = 0
    for (feature_set, scaler), (X, feature_names, transformer) in matrices.items():
        for k in k_values:
            model = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
            labels = model.fit_predict(X)
            metrics = compute_metrics(X, labels, random_seed=random_seed)
            rows.append(
                {
                    "model_id": model_id,
                    "algorithm": "kmeans",
                    "feature_set": feature_set,
                    "scaler": scaler,
                    "params": f"n_clusters={k}",
                    "sample_size": len(X),
                    "inertia": float(model.inertia_),
                    "selected": False,
                    **metrics,
                }
            )
            models[model_id] = ModelResult(
                estimator=model,
                transformer=transformer,
                labels=labels,
                feature_names=feature_names,
                metadata={"algorithm": "kmeans", "feature_set": feature_set, "scaler": scaler, "params": {"n_clusters": k}},
            )
            model_id += 1
    return pd.DataFrame(rows), models


def run_agglomerative_grid(
    matrices: dict[tuple[str, str], tuple[np.ndarray, list[str], Any]],
    k_values: range,
    random_seed: int = 42,
    max_rows: int = 3000,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_seed)
    linkages = ["complete", "average"]
    for (feature_set, scaler), (X, _, _) in matrices.items():
        sample_idx = rng.choice(len(X), size=min(len(X), max_rows), replace=False)
        X_sample = X[sample_idx]
        for linkage_name in linkages:
            for k in k_values:
                model = HierarchicalClustering(n_clusters=k, linkage=linkage_name)
                labels = model.fit_predict(X_sample)
                metrics = compute_metrics(X_sample, labels, random_seed=random_seed)
                rows.append(
                    {
                        "model_id": None,
                        "algorithm": "agglomerative",
                        "feature_set": feature_set,
                        "scaler": scaler,
                        "params": f"n_clusters={k}, linkage={linkage_name}",
                        "sample_size": len(X_sample),
                        "inertia": None,
                        "selected": False,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def run_dbscan_grid(
    matrices: dict[tuple[str, str], tuple[np.ndarray, list[str], Any]],
    eps_values: list[float],
    min_samples_values: list[int],
    random_seed: int = 42,
    max_rows: int = 5000,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_seed)
    for (feature_set, scaler), (X, _, _) in matrices.items():
        sample_idx = rng.choice(len(X), size=min(len(X), max_rows), replace=False)
        X_sample = X[sample_idx]
        for eps in eps_values:
            for min_samples in min_samples_values:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_sample)
                metrics = compute_metrics(X_sample, labels, random_seed=random_seed)
                rows.append(
                    {
                        "model_id": None,
                        "algorithm": "dbscan",
                        "feature_set": feature_set,
                        "scaler": scaler,
                        "params": f"eps={eps}, min_samples={min_samples}",
                        "sample_size": len(X_sample),
                        "inertia": None,
                        "selected": False,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def select_best_model(metrics: pd.DataFrame) -> pd.Series:
    candidates = metrics.copy()
    candidates = candidates[candidates["algorithm"] == "kmeans"].copy()
    candidates = candidates[candidates["n_clusters"].between(3, 7)]
    candidates = candidates[candidates["silhouette"].notna()]
    candidates["feature_bonus"] = candidates["feature_set"].map(
        {"income_spending_only": 0.06, "behavior_numeric": 0.04, "age_spending_only": 0.02, "behavior_plus_gender": 0.0}
    ).fillna(0.0)
    candidates["selection_score"] = candidates["silhouette"] + candidates["feature_bonus"]
    return candidates.sort_values(
        ["selection_score", "silhouette"],
        ascending=[False, False],
    ).iloc[0]


def assign_cluster_labels(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    result = df.copy()
    result["cluster"] = labels.astype(int)
    return result


def _level(value: float, low: float, high: float) -> str:
    if value <= low:
        return "Low"
    if value >= high:
        return "High"
    return "Moderate"


def _segment_name(row: pd.Series, quantiles: dict[str, tuple[float, float]]) -> str:
    income = _level(row["annual_income_mean"], *quantiles["annual_income"])
    spend = _level(row["spending_score_mean"], *quantiles["spending_score"])
    age = _level(row["age_mean"], *quantiles["age"])
    if income == "High" and spend == "High":
        return "High income, high spending"
    if income == "High" and spend == "Low":
        return "High income, low spending"
    if income == "Low" and spend == "High":
        return "Low income, high spending"
    if income == "Low" and spend == "Low":
        return "Low income, low spending"
    if age == "Low" and spend == "High":
        return "Younger high spenders"
    if age == "High" and spend == "Low":
        return "Older conservative spenders"
    return f"{income} income, {spend.lower()} spending"


def _business_action(name: str) -> str:
    actions = {
        "High income, high spending": "Prioritize loyalty perks, premium offers, and early access campaigns.",
        "High income, low spending": "Use personalized recommendations and targeted incentives to lift conversion.",
        "Low income, high spending": "Offer bundles, discounts, and value-focused retention campaigns.",
        "Low income, low spending": "Use low-cost engagement and avoid expensive acquisition-style campaigns.",
        "Younger high spenders": "Promote trend-led launches, social campaigns, and limited-time offers.",
        "Older conservative spenders": "Use practical offers, trust-building messaging, and measured reactivation.",
    }
    return actions.get(name, "Use broad retention campaigns, seasonal offers, and monitor movement between segments.")


def profile_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    total = len(clustered)
    grouped = clustered.groupby("cluster")
    profile = grouped.agg(
        count=("customer_id", "count"),
        age_mean=("age", "mean"),
        age_median=("age", "median"),
        annual_income_mean=("annual_income", "mean"),
        annual_income_median=("annual_income", "median"),
        spending_score_mean=("spending_score", "mean"),
        spending_score_median=("spending_score", "median"),
    ).reset_index()
    profile["percentage"] = profile["count"] / total
    gender_dist = (
        clustered.pivot_table(index="cluster", columns="gender", values="customer_id", aggfunc="count", fill_value=0)
        .div(grouped.size(), axis=0)
        .add_prefix("gender_share_")
        .reset_index()
    )
    profile = profile.merge(gender_dist, on="cluster", how="left")
    quantiles = {
        "age": tuple(clustered["age"].quantile([0.33, 0.67]).to_numpy()),
        "annual_income": tuple(clustered["annual_income"].quantile([0.33, 0.67]).to_numpy()),
        "spending_score": tuple(clustered["spending_score"].quantile([0.33, 0.67]).to_numpy()),
    }
    profile["segment_name"] = profile.apply(lambda row: _segment_name(row, quantiles), axis=1)
    counts: dict[str, int] = {}
    unique_names = []
    for name in profile["segment_name"]:
        counts[name] = counts.get(name, 0) + 1
        unique_names.append(name if counts[name] == 1 else f"{name} ({counts[name]})")
    profile["segment_name"] = unique_names
    profile["recommended_business_action"] = profile["segment_name"].str.replace(r" \(\d+\)$", "", regex=True).map(_business_action)
    numeric_columns = profile.select_dtypes(include="number").columns
    profile[numeric_columns] = profile[numeric_columns].round(4)
    return profile.sort_values("cluster").reset_index(drop=True)


def plot_kmeans_diagnostics(metrics: pd.DataFrame, output_dir: Path, feature_set: str, scaler: str) -> list[Path]:
    subset = metrics[
        (metrics["algorithm"] == "kmeans") & (metrics["feature_set"] == feature_set) & (metrics["scaler"] == scaler)
    ].sort_values("n_clusters")
    paths = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=subset, x="n_clusters", y="inertia", marker="o", ax=ax, color="#4C78A8")
    ax.set_title("K-Means Elbow Curve")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    paths.append(save_figure(fig, output_dir / "kmeans_elbow_curve.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=subset, x="n_clusters", y="silhouette", marker="o", ax=ax, color="#F58518")
    ax.set_title("K-Means Silhouette Curve")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    paths.append(save_figure(fig, output_dir / "kmeans_silhouette_curve.png"))
    return paths


def plot_cluster_scatter(clustered: pd.DataFrame, output_dir: Path, x_col: str, y_col: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=clustered,
        x=x_col,
        y=y_col,
        hue="cluster",
        palette="tab10",
        alpha=0.5,
        s=22,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(f"Final Clusters: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}")
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    return save_figure(fig, output_dir / f"clusters_{x_col}_vs_{y_col}.png")


def plot_pca_projection(X: np.ndarray, labels: np.ndarray, output_dir: Path) -> Path:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plot_df = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": labels})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="cluster", palette="tab10", alpha=0.5, s=22, linewidth=0, ax=ax)
    ax.set_title("PCA Projection Of Final Clusters")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    return save_figure(fig, output_dir / "clusters_pca_projection.png")


def plot_cluster_centers(centers: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = centers.reset_index(names="cluster").melt(id_vars="cluster", var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="cluster", y="value", hue="feature", ax=ax)
    ax.set_title("K-Means Cluster Centers In Original Scale")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature value")
    return save_figure(fig, output_dir / "cluster_centers_original_scale.png")


def plot_dendrogram_sample(X: np.ndarray, output_dir: Path, random_seed: int = 42, max_rows: int = 80) -> Path:
    rng = np.random.default_rng(random_seed)
    sample_idx = rng.choice(len(X), size=min(len(X), max_rows), replace=False)
    linked = linkage(X[sample_idx], method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, truncate_mode="level", p=5, ax=ax)
    ax.set_title("Agglomerative Dendrogram Sample")
    ax.set_xlabel("Sampled customers")
    ax.set_ylabel("Distance")
    return save_figure(fig, output_dir / "agglomerative_dendrogram_sample.png")


def save_final_model(model_result: ModelResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": model_result.estimator,
            "transformer": model_result.transformer,
            "feature_names": model_result.feature_names,
            "metadata": model_result.metadata,
        },
        path,
    )
    return path

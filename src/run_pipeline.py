from __future__ import annotations

from pathlib import Path

import pandas as pd

import clustering
import eda
import preprocess
import utils


RANDOM_SEED = 42
PROJECT_ROOT = utils.project_root()


def feature_numeric_columns(feature_set: str) -> list[str]:
    if feature_set == "income_spending_only":
        return ["annual_income", "spending_score"]
    if feature_set == "age_spending_only":
        return ["age", "spending_score"]
    return ["age", "annual_income", "spending_score"]


def write_customer_report(
    path: Path,
    selected: pd.Series,
    profiles: pd.DataFrame,
    metrics: pd.DataFrame,
    raw_path: Path,
) -> Path:
    top_metrics = (
        metrics[metrics["algorithm"].isin(["kmeans", "agglomerative", "dbscan"])]
        .sort_values(["selected", "silhouette"], ascending=[False, False])
        .head(12)
    )
    content = [
        "# Customer Segmentation Report",
        "",
        "## Dataset",
        "",
        f"- Raw file: `{raw_path}`",
        "- The dataset contains customer ID, gender, age, annual income, and spending score.",
        "- Customer ID is preserved for traceability but excluded from clustering.",
        "- Gender is compared through a one-hot encoded experiment rather than ordinal encoding.",
        "",
        "## Selected Model",
        "",
        f"- Algorithm: `{selected['algorithm']}`",
        f"- Feature set: `{selected['feature_set']}`",
        f"- Scaler: `{selected['scaler']}`",
        f"- Parameters: `{selected['params']}`",
        f"- Number of clusters: `{int(selected['n_clusters'])}`",
        f"- Silhouette: `{selected['silhouette']:.4f}`",
        "",
        "The selected solution balances quantitative quality with interpretability. "
        "The final choice prefers 3 to 7 clusters, strong silhouette performance, "
        "and segments that are readable in income/spending plots.",
        "",
        "## Segment Profiles",
        "",
        profiles.to_markdown(index=False),
        "",
        "## Model Comparison Snapshot",
        "",
        top_metrics[
            [
                "algorithm",
                "feature_set",
                "scaler",
                "params",
                "n_clusters",
                "noise_rate",
                "silhouette",
                "selected",
            ]
        ].to_markdown(index=False),
        "",
        "## Limitations",
        "",
        "- Clusters describe behavioral similarity, not causal drivers of spending.",
        "- Annual income appears to be measured in full currency units rather than `k$`; interpretation uses relative income bands.",
        "- Agglomerative clustering and DBSCAN are evaluated on deterministic samples to keep runtime practical for this dataset size.",
        "- Segment names are business-friendly summaries of cluster averages and should be validated with domain context before campaign launch.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def main() -> None:
    utils.set_random_seed(RANDOM_SEED)
    paths = utils.ensure_directories(PROJECT_ROOT)
    config = utils.load_config(PROJECT_ROOT)
    raw_path = utils.find_raw_csv(PROJECT_ROOT, config["raw_data_glob"])

    utils.status(f"Loading raw data from {raw_path}")
    raw_df = preprocess.load_csv(raw_path)
    normalized_raw = preprocess.normalize_column_names(raw_df)
    clean_df = preprocess.clean_customer_data(raw_df)

    missing_summary = preprocess.missing_value_summary(normalized_raw)
    duplicate_checks = preprocess.duplicate_summary(normalized_raw)
    invalid_checks = preprocess.invalid_value_summary(normalized_raw)
    outliers = preprocess.outlier_summary(clean_df)
    eda_summary = preprocess.make_eda_summary(clean_df)

    utils.save_table(eda_summary, paths["metrics"] / "eda_summary.csv")
    utils.save_table(missing_summary, paths["metrics"] / "missing_values.csv")
    utils.save_table(duplicate_checks, paths["metrics"] / "duplicate_summary.csv")
    utils.save_table(invalid_checks, paths["metrics"] / "invalid_value_summary.csv")
    utils.save_table(outliers, paths["metrics"] / "outlier_summary.csv")
    utils.save_table(clean_df, paths["processed"] / "customers_clean.csv")

    preprocess.write_preprocessing_report(
        paths["metrics"] / "preprocessing_report.md",
        raw_df,
        clean_df,
        missing_summary,
        duplicate_checks,
        invalid_checks,
        outliers,
    )

    utils.status("Saving EDA figures")
    eda.plot_missing_values(normalized_raw, paths["figures_before"])
    eda.plot_duplicate_summary(duplicate_checks, paths["figures_before"])
    eda.plot_boxplots(clean_df, paths["figures_before"], suffix="before")
    eda.plot_distributions(clean_df, paths["figures_before"], suffix="before")
    eda.plot_gender_distribution(clean_df, paths["figures_before"], suffix="before")
    eda.plot_relationships(clean_df, paths["figures_before"], suffix="before")
    eda.plot_pairplot(clean_df, paths["figures_before"], suffix="before")
    eda.plot_correlation_heatmap(clean_df, paths["figures_before"], suffix="before")

    matrices = {}
    scaled_output = clean_df[["customer_id"]].copy()
    for feature_set in config["feature_sets"]:
        for scaler in config["scalers"]:
            X, feature_names, transformer = preprocess.build_feature_matrix(clean_df, feature_set=feature_set, scaler_name=scaler)
            matrices[(feature_set, scaler)] = (X, feature_names, transformer)
            if feature_set == "behavior_numeric" and scaler == "standard":
                for idx, name in enumerate(feature_names):
                    scaled_output[f"scaled_{name}"] = X[:, idx]
    utils.save_table(scaled_output, paths["processed"] / "customers_scaled.csv")

    utils.status("Running clustering experiments")
    k_start, k_end = config["kmeans_k_range"]
    kmeans_metrics, kmeans_models = clustering.run_kmeans_grid(matrices, range(k_start, k_end + 1), random_seed=RANDOM_SEED)

    alt_keys = [
        ("behavior_numeric", "standard"),
        ("income_spending_only", "standard"),
        ("behavior_plus_gender", "standard"),
    ]
    alt_matrices = {key: matrices[key] for key in alt_keys if key in matrices}
    a_start, a_end = config["agglomerative_k_range"]
    agglomerative_metrics = clustering.run_agglomerative_grid(
        alt_matrices,
        range(a_start, a_end + 1),
        random_seed=RANDOM_SEED,
        max_rows=1000,
    )
    dbscan_metrics = clustering.run_dbscan_grid(
        alt_matrices,
        config["dbscan_eps_values"],
        config["dbscan_min_samples"],
        random_seed=RANDOM_SEED,
        max_rows=2000,
    )

    metrics = pd.concat([kmeans_metrics, agglomerative_metrics, dbscan_metrics], ignore_index=True)
    selected = clustering.select_best_model(metrics)
    selected_id = int(selected["model_id"])
    metrics.loc[metrics["model_id"] == selected_id, "selected"] = True
    utils.save_table(metrics, paths["metrics"] / "clustering_metrics.csv")

    selected_result = kmeans_models[selected_id]
    clustered = clustering.assign_cluster_labels(clean_df, selected_result.labels)
    profiles = clustering.profile_clusters(clustered)
    utils.save_table(clustered, paths["processed"] / "clustering_results.csv")
    utils.save_table(profiles, paths["metrics"] / "segment_profiles.csv")

    utils.status("Saving final clustering figures and model")
    clustering.plot_kmeans_diagnostics(
        metrics,
        paths["figures_clustering"],
        selected_result.metadata["feature_set"],
        selected_result.metadata["scaler"],
    )
    clustering.plot_cluster_scatter(clustered, paths["figures_clustering"], "annual_income", "spending_score")
    clustering.plot_cluster_scatter(clustered, paths["figures_clustering"], "age", "spending_score")
    clustering.plot_cluster_scatter(clustered, paths["figures_clustering"], "age", "annual_income")
    X_selected, _, _ = matrices[(selected_result.metadata["feature_set"], selected_result.metadata["scaler"])]
    clustering.plot_pca_projection(X_selected, selected_result.labels, paths["figures_clustering"])
    clustering.plot_dendrogram_sample(X_selected, paths["figures_clustering"], random_seed=RANDOM_SEED)

    if selected_result.metadata["algorithm"] == "kmeans":
        numeric_columns = feature_numeric_columns(selected_result.metadata["feature_set"])
        centers = preprocess.inverse_numeric_centers(selected_result.estimator.cluster_centers_, selected_result.transformer, numeric_columns)
        utils.save_table(centers.reset_index(names="cluster"), paths["metrics"] / "cluster_centers.csv")
        clustering.plot_cluster_centers(centers, paths["figures_clustering"])

    clustering.save_final_model(selected_result, paths["models"] / "best_cluster_model.joblib")
    write_customer_report(paths["reports"] / "customer_segmentation_report.md", selected, profiles, metrics, raw_path)

    utils.status("Pipeline complete")
    print(
        {
            "selected_algorithm": selected["algorithm"],
            "selected_feature_set": selected["feature_set"],
            "selected_scaler": selected["scaler"],
            "selected_params": selected["params"],
            "n_clusters": int(selected["n_clusters"]),
            "silhouette": round(float(selected["silhouette"]), 4),
        }
    )


if __name__ == "__main__":
    main()

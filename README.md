# Lab 03 - Customer Segmentation

This project clusters retail customers into business-friendly segments using age, annual income, spending score, and optional gender features. It follows the implementation guide in `agents/codex.md`.

## Dataset

Raw data is stored at:

```text
data/raw/Shopping Mall Customer Segmentation Data .csv
```

The pipeline normalizes the raw columns into:

- `customer_id`
- `age`
- `gender`
- `annual_income`
- `spending_score`

The current dataset has 15,079 rows, no missing values, and no duplicate rows.

## Project Structure

```text
lab03/
  README.md
  requirements.txt
  agents/
    codex.md
  config/
    clustering_config.json
  data/
    raw/
    processed/
      metrics/
  notebooks/
    lab03.ipynb
    utils.py
    preprocess.py
    eda.py
    clustering.py
    run_pipeline.py
  reports/
    figures/
      before_process/
      after_process/
      clustering/
    customer_segmentation_report.md
  models/
    best_cluster_model.joblib
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run The Full Pipeline

```bash
.venv/bin/python notebooks/run_pipeline.py
```

This generates:

- `data/processed/customers_clean.csv`
- `data/processed/customers_scaled.csv`
- `data/processed/clustering_results.csv`
- `data/processed/metrics/eda_summary.csv`
- `data/processed/metrics/clustering_metrics.csv`
- `data/processed/metrics/segment_profiles.csv`
- `data/processed/metrics/preprocessing_report.md`
- `reports/figures/before_process/*.png`
- `reports/figures/clustering/*.png`
- `reports/customer_segmentation_report.md`
- `models/best_cluster_model.joblib`

## Notebook

Open and run:

```text
notebooks/lab03.ipynb
```

The notebook is structured as a report:

1. Imports and setup
2. Raw data overview
3. Data quality checks
4. EDA before preprocessing
5. Preprocessing and scaling
6. Clustering experiments
7. Model selection
8. Cluster visualization
9. Segment profiling
10. Artifact saving

## Current Result

The reproducible pipeline currently selects:

- Algorithm: K-Means
- Feature set: `income_spending_only`
- Scaler: `standard`
- Clusters: `4`
- Silhouette score: about `0.408`

The final segment table is saved at:

```text
data/processed/metrics/segment_profiles.csv
```

## Notes

- `Customer ID` is never used as a clustering feature.
- `Gender` is not ordinal encoded. It is only included in one-hot encoded feature-set experiments.
- K-Means is evaluated on the full dataset.
- Agglomerative clustering and DBSCAN are evaluated on deterministic samples to keep runtime practical for 15k rows.
- Segment names are descriptive business summaries, not causal claims.

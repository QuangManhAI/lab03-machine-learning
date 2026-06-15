# Codex Guide - Lab 03 Customer Segmentation

## Project Goal

Build a complete customer segmentation project for a retail customer dataset with these fields:

- `CustomerID` or similar identifier
- `Gender`
- `Age`
- `Annual Income (k$)` or annual income equivalent
- `Spending Score (1-100)` or spending score equivalent

The final project should mirror the structure and quality of the email spam project in `lab01`: clear data flow, reproducible preprocessing, rich EDA, multiple model comparisons, quantitative evaluation, visual diagnostics, and a notebook that tells the story from raw data to final customer segments.

Primary objective:

> Cluster customers into meaningful segments and explain each segment in business-friendly language.

Do not stop at fitting K-Means. The project must explain why the chosen clustering setup is reasonable, how it compares with alternatives, and what each segment means.

## Expected Repository Structure

Create or maintain this structure:

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
      customers.csv
    processed/
      customers_clean.csv
      customers_scaled.csv
      clustering_results.csv
      metrics/
        eda_summary.csv
        clustering_metrics.csv
        segment_profiles.csv
        preprocessing_report.md
  notebooks/
    lab03.ipynb
    utils.py
    preprocess.py
    eda.py
    clustering.py
  reports/
    figures/
      before_process/
      after_process/
      clustering/
    customer_segmentation_report.md
  models/
    best_cluster_model.joblib
```

If the dataset uses a different file name, keep the code flexible and document the actual path in the notebook.

## Implementation Principles

- Prefer small reusable Python modules over putting all logic directly in notebook cells.
- The notebook should be readable as a report, with markdown explaining each step.
- Every transformation should be reproducible with a fixed random seed.
- Never scale ID columns such as `CustomerID`.
- Do not use `Gender` blindly as an ordinal numeric feature. Either one-hot encode it for clustering experiments or exclude it from the main numeric baseline and compare both choices.
- Keep raw data unchanged. Write cleaned/scaled outputs under `data/processed/`.
- Save important figures under `reports/figures/`.
- Save metric tables under `data/processed/metrics/`.

## Notebook Flow

Build `notebooks/lab03.ipynb` with this narrative:

### Step 0 - Imports And Setup

- Define `PROJECT_ROOT`.
- Import local modules: `utils`, `preprocess`, `eda`, `clustering`.
- Set `RANDOM_SEED = 42`.
- Print module paths so it is obvious the notebook is using local code.

### Step 1 - Load Data And Raw Overview

Load the customer dataset and show:

- shape
- column names and dtypes
- first rows
- missing values per column
- duplicate rows
- basic descriptive statistics
- count of unique values for categorical columns

Expected outputs:

- `data/processed/metrics/eda_summary.csv`
- raw overview tables displayed in notebook

### Step 2 - Data Quality Checks

Analyze:

- missing values
- duplicate rows
- invalid ages or income values
- invalid spending scores outside expected range
- outliers using IQR and z-score summaries

Important:

- Do not remove outliers automatically without showing their impact.
- For this lab, keep a "clean baseline" and optionally compare a "winsorized/robust" variant if outliers are meaningful.

Expected figures:

- missing value chart
- duplicate summary chart
- boxplots for `Age`, `Annual Income`, `Spending Score`

### Step 3 - EDA Before Preprocessing

Visualize distributions and relationships:

- histograms/KDE plots for age, income, spending score
- gender distribution if available
- scatter: income vs spending score
- scatter: age vs spending score
- scatter: age vs income
- pairplot or scatter matrix for numeric features
- correlation heatmap for numeric features

The notebook should explain early visual patterns, especially the classic income/spending groups if present.

Expected figure directory:

```text
reports/figures/before_process/
```

### Step 4 - Preprocessing And Feature Scaling

Create reusable preprocessing functions:

- normalize column names into safe internal names:
  - `customer_id`
  - `gender`
  - `age`
  - `annual_income`
  - `spending_score`
- validate required columns
- clean missing/invalid values
- encode gender only for experiments that include it
- scale numeric features using:
  - `StandardScaler`
  - `MinMaxScaler`
  - optionally `RobustScaler`

Run at least two feature sets:

1. `behavior_numeric`
   - `age`
   - `annual_income`
   - `spending_score`

2. `behavior_plus_gender`
   - scaled numeric features
   - one-hot gender features

Optional feature sets:

- `income_spending_only`
- `age_spending_only`

Expected outputs:

- `data/processed/customers_clean.csv`
- `data/processed/customers_scaled.csv`
- `data/processed/metrics/preprocessing_report.md`

### Step 5 - Clustering Experiments

Implement and compare multiple clustering algorithms.

Required baseline:

- K-Means

Recommended comparisons:

- Agglomerative clustering
- DBSCAN
- K-Medoids if dependency is available; otherwise document why skipped

For K-Means:

- test `k = 2..10`
- use elbow method with inertia
- use silhouette coefficient
- use Calinski-Harabasz score
- use Davies-Bouldin score
- keep random seed fixed

For Agglomerative:

- test `n_clusters = 2..10`
- test linkage options where valid:
  - `ward`
  - `complete`
  - `average`

For DBSCAN:

- test several `eps` and `min_samples`
- report number of clusters excluding noise
- report noise rate
- avoid selecting a DBSCAN result that labels most points as noise

Expected metrics table:

```text
data/processed/metrics/clustering_metrics.csv
```

Suggested columns:

- `algorithm`
- `feature_set`
- `scaler`
- `params`
- `n_clusters`
- `noise_rate`
- `silhouette`
- `calinski_harabasz`
- `davies_bouldin`
- `inertia`
- `selected`

### Step 6 - Model Selection

Select the final clustering result using a balanced decision, not one metric alone.

Default selection rule:

1. Prefer models with `n_clusters` between 3 and 7 for interpretability.
2. Prefer higher silhouette.
3. Prefer lower Davies-Bouldin.
4. Prefer a visible, explainable segmentation in income/spending scatter plots.
5. Avoid solutions with tiny unusable clusters unless clearly meaningful.

Expected final model candidate:

- Usually K-Means with `k=5` on `annual_income + spending_score` or `age + annual_income + spending_score`, depending on the data.

Do not hard-code `k=5` without evaluation. If `k=5` wins, show why.

### Step 7 - Cluster Visualization

Visualize the final model:

- scatter plot: annual income vs spending score, colored by cluster
- scatter plot: age vs spending score, colored by cluster
- scatter plot: age vs annual income, colored by cluster
- PCA 2D projection colored by cluster
- optional dendrogram for hierarchical clustering

For K-Means:

- show cluster centers in original feature scale where possible
- annotate or summarize each cluster

Expected figure directory:

```text
reports/figures/clustering/
```

### Step 8 - Segment Profiling

Create a segment profile table with:

- cluster id
- row count
- percentage of customers
- mean/median age
- mean/median income
- mean/median spending score
- gender distribution if available
- plain-English segment name
- recommended business action

Example segment names:

- `High income, high spending`
- `High income, low spending`
- `Low income, high spending`
- `Low income, low spending`
- `Moderate income, moderate spending`
- `Younger high spenders`
- `Older conservative spenders`

Expected output:

```text
data/processed/metrics/segment_profiles.csv
data/processed/clustering_results.csv
```

### Step 9 - Evaluation Summary

The notebook should explicitly answer:

- Which algorithm was selected?
- Which feature set was selected?
- Which scaler was selected?
- How many clusters were selected?
- What are the silhouette, Calinski-Harabasz, and Davies-Bouldin scores?
- What are the segment profiles?
- What business actions are suggested for each segment?
- What are the limitations?

### Step 10 - Save Artifacts

Save:

- final model: `models/best_cluster_model.joblib`
- scaler/encoder if needed
- clustered dataset
- metrics tables
- figures
- report markdown

## Python Module Responsibilities

### `notebooks/utils.py`

General project helpers:

- path setup
- display helpers
- save table helpers
- save figure helpers
- random seed setup
- report status helper

### `notebooks/preprocess.py`

Data cleaning and transformation:

- load CSV
- normalize column names
- validate columns
- missing-value summary
- duplicate summary
- outlier summary
- clean customer data
- build feature matrices
- scale features
- inverse-transform cluster centers

### `notebooks/eda.py`

Exploration and visualization:

- distributions
- boxplots
- countplots
- scatter plots
- pairplot/scatter matrix
- correlation heatmap
- before/after preprocessing comparisons

### `notebooks/clustering.py`

Clustering experiments:

- run K-Means grid
- run Agglomerative grid
- run DBSCAN grid
- compute metrics
- select best model
- assign cluster labels
- profile clusters
- plot clusters
- plot elbow and silhouette curves
- save final model

## Metrics Guidance

Use these metrics carefully:

- Silhouette coefficient: higher is better, range roughly `[-1, 1]`.
- Calinski-Harabasz: higher is better.
- Davies-Bouldin: lower is better.
- K-Means inertia: lower always decreases as `k` increases, so use it only for elbow analysis.

For DBSCAN:

- Silhouette should be computed only when at least 2 non-noise clusters exist.
- Report noise rate.
- If too many points are noise, mark the configuration as weak even if a metric looks good.

## Visual Design Requirements

Figures should be clean and report-ready:

- clear titles
- labeled axes
- legends with cluster names or cluster ids
- readable colors
- saved with consistent sizes
- no overlapping labels

Use the same visual discipline as `lab01`: each figure should answer a question, not just decorate the notebook.

## Business Interpretation Requirements

The final report must translate clusters into retail actions.

Examples:

- High income + high spending:
  - loyalty program
  - premium offers
  - early access campaigns

- High income + low spending:
  - targeted incentives
  - personalized recommendations
  - conversion campaigns

- Low income + high spending:
  - discounts
  - bundles
  - value-focused promotions

- Low income + low spending:
  - low-cost engagement
  - avoid expensive campaigns

- Moderate customers:
  - broad retention campaign
  - seasonal offers

Avoid overclaiming. Clusters are behavioral segments, not causal explanations.

## Acceptance Criteria

The project is complete when:

- The notebook runs from top to bottom without errors.
- Raw and processed datasets are separated.
- Missing values, duplicates, outliers, and distributions are analyzed.
- Numeric features are scaled before clustering.
- At least K-Means and one alternative algorithm are evaluated.
- K-Means includes elbow and silhouette analysis for multiple `k`.
- A final clustering model is selected with a clear reason.
- Cluster plots are saved and displayed.
- Segment profile table exists and has human-readable segment names.
- Final report summarizes findings and business recommendations.
- All generated artifacts are saved in the expected folders.

## Recommended Development Order For Codex

1. Inspect current files and dataset availability.
2. Create folder structure.
3. Add requirements if missing:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `matplotlib`
   - `seaborn`
   - `joblib`
   - optional: `scikit-learn-extra` for K-Medoids
4. Implement `preprocess.py`.
5. Implement `eda.py`.
6. Implement `clustering.py`.
7. Implement `utils.py`.
8. Build `lab03.ipynb` as a polished report notebook.
9. Run the notebook or equivalent script end to end.
10. Fix any broken plots, paths, or metric edge cases.
11. Update `README.md` with how to run the project.

## Notes For Future Codex Runs

- If no dataset is present, create clear instructions in `README.md` for placing the CSV at `data/raw/customers.csv`.
- If a common Mall Customers dataset is used, map:
  - `Genre` or `Gender` -> `gender`
  - `Age` -> `age`
  - `Annual Income (k$)` -> `annual_income`
  - `Spending Score (1-100)` -> `spending_score`
- Keep implementation deterministic with `random_state=42`.
- Prefer explaining trade-offs over chasing a single best score.
- The final deliverable should feel like a complete customer segmentation analysis, not a one-cell clustering demo.

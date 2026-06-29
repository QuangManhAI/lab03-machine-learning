Input: -
Output: -
To do: - 

- https://www.kaggle.com/datasets/zubairmustafa/shopping-mall-customer-segmentation-data

*notebook structure*: create and optimize notebook **lab03.ipynb** based on codex.md
struct of notebook: 
- cell md: Overview all steps in notebook. Draw a table containing: Step Number, Step Name, Description, Python Modules Used, and Saved Outputs/Metrics.
- cell md: Step 0 — Import Libraries And Setup
- cell code: Import libraries, local modules (utils, preprocess, eda, clustering), set RANDOM_SEED = 42, and print module paths.
- cell md: STEP 1 — Load Data And Raw Overview
- cell code: Load customer dataset, show shape, dtypes, first rows, missing values, duplicates, and descriptive statistics. Export raw overview metrics to `data/processed/metrics/eda_summary.csv`.
- cell md: STEP 2 — Data Quality Checks
- cell code: Analyze missing values, duplicates, invalid ages/income/spending scores, and outline outlier z-score/IQR summaries. Plot missing values, duplicates, and boxplots for Age, Annual Income, and Spending Score.
- cell md: STEP 3 — EDA Before Preprocessing
- cell code: Plot distributions (histograms/KDE), gender ratios, correlation heatmap, and scatter plots (income vs spending score, age vs spending score, age vs income) under `reports/figures/before_process/`.
- cell md: STEP 4 — Preprocessing And Feature Scaling
- cell code: Normalize column names (customer_id, gender, age, annual_income, spending_score), clean invalid entries, scale features using StandardScaler/MinMaxScaler, and build feature sets (behavior_numeric, behavior_plus_gender). Save clean/scaled datasets and generate `data/processed/metrics/preprocessing_report.md`.
- cell md: STEP 5 — Clustering Experiments
- cell code: Run grid experiments for K-Means (inertia, silhouette, CH, DB scores for k=2..10), Agglomerative (n_clusters=2..10 with ward/complete/average linkages), and DBSCAN (eps/min_samples tuning, reporting noise rate). Save metrics table `data/processed/metrics/clustering_metrics.csv`.
- cell md: STEP 6 — Model Selection
- cell code: Filter model configurations, compare silhouette vs Davies-Bouldin vs cluster count (3 to 7 clusters), pick the best model, and write code to explain the final chosen strategy, scaler, and features.
- cell md: STEP 7 — Cluster Visualization
- cell code: Plot final cluster scatter plots (income vs spending, age vs spending, age vs income) and PCA 2D projections. Annotate cluster centroids and save figures to `reports/figures/clustering/`.
- cell md: STEP 8 — Segment Profiling
- cell code: Create profiles aggregating cluster size, mean/median age, income, and spending. Save profiles to `data/processed/metrics/segment_profiles.csv` and label the original dataset to save under `data/processed/clustering_results.csv`.
- cell md: STEP 9 — Evaluation Summary & Business Interpretation
- cell md: Summarize the chosen model details (scaler, cluster count, scores), segment name profiles (e.g., Younger high spenders), retail business recommendations, and limitations.
- cell md: STEP 10 — Save Artifacts & Deployment Check
- cell code: Save the selected best model to `models/best_cluster_model.joblib`. Verify the saved model by loading it back and testing predictions on dummy customer data.

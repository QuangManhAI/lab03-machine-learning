# Customer Segmentation Report

## Dataset

- Raw file: `/Users/quangmanh/Project/lab03/data/raw/Shopping Mall Customer Segmentation Data .csv`
- The dataset contains customer ID, gender, age, annual income, and spending score.
- Customer ID is preserved for traceability but excluded from clustering.
- Gender is compared through a one-hot encoded experiment rather than ordinal encoding.

## Selected Model

- Algorithm: `kmeans`
- Feature set: `income_spending_only`
- Scaler: `minmax`
- Parameters: `n_clusters=4`
- Number of clusters: `4`
- Silhouette: `0.4080`

The selected solution balances quantitative quality with interpretability. The final choice prefers 3 to 7 clusters, strong silhouette performance, and segments that are readable in income/spending plots.

## Segment Profiles

|   cluster |   count |   age_mean |   age_median |   annual_income_mean |   annual_income_median |   spending_score_mean |   spending_score_median |   percentage |   gender_share_Female |   gender_share_Male | segment_name               | recommended_business_action                                                  |
|----------:|--------:|-----------:|-------------:|---------------------:|-----------------------:|----------------------:|------------------------:|-------------:|----------------------:|--------------------:|:---------------------------|:-----------------------------------------------------------------------------|
|         0 |    3646 |    54.1777 |           54 |             156296   |               156434   |               25.5702 |                      25 |       0.2418 |                0.506  |              0.494  | High income, low spending  | Use personalized recommendations and targeted incentives to lift conversion. |
|         1 |    3865 |    54.2561 |           54 |              65182.9 |                64463   |               25.7946 |                      26 |       0.2563 |                0.4903 |              0.5097 | Low income, low spending   | Use low-cost engagement and avoid expensive acquisition-style campaigns.     |
|         2 |    3782 |    53.9614 |           54 |              64892.4 |                65109.5 |               75.5153 |                      76 |       0.2508 |                0.4952 |              0.5048 | Low income, high spending  | Offer bundles, discounts, and value-focused retention campaigns.             |
|         3 |    3786 |    54.369  |           55 |             155204   |               155002   |               75.1049 |                      75 |       0.2511 |                0.4942 |              0.5058 | High income, high spending | Prioritize loyalty perks, premium offers, and early access campaigns.        |

## Model Comparison Snapshot

| algorithm   | feature_set          | scaler   | params                   |   n_clusters |   noise_rate |   silhouette | selected   |
|:------------|:---------------------|:---------|:-------------------------|-------------:|-------------:|-------------:|:-----------|
| kmeans      | income_spending_only | minmax   | n_clusters=4             |            4 |       0      |     0.407964 | True       |
| dbscan      | behavior_numeric     | standard | eps=0.25, min_samples=10 |            3 |       0.982  |     0.834625 | False      |
| dbscan      | behavior_plus_gender | standard | eps=0.25, min_samples=5  |           54 |       0.833  |     0.628384 | False      |
| kmeans      | behavior_plus_gender | minmax   | n_clusters=2             |            2 |       0      |     0.57873  | False      |
| dbscan      | behavior_plus_gender | standard | eps=0.35, min_samples=10 |           19 |       0.8835 |     0.543398 | False      |
| dbscan      | behavior_plus_gender | standard | eps=0.5, min_samples=20  |           13 |       0.764  |     0.461684 | False      |
| dbscan      | behavior_numeric     | standard | eps=0.5, min_samples=35  |            5 |       0.792  |     0.433079 | False      |
| kmeans      | income_spending_only | standard | n_clusters=4             |            4 |       0      |     0.407961 | False      |
| kmeans      | income_spending_only | robust   | n_clusters=4             |            4 |       0      |     0.407806 | False      |
| kmeans      | age_spending_only    | minmax   | n_clusters=4             |            4 |       0      |     0.406928 | False      |
| kmeans      | age_spending_only    | robust   | n_clusters=4             |            4 |       0      |     0.406905 | False      |
| kmeans      | age_spending_only    | standard | n_clusters=4             |            4 |       0      |     0.406904 | False      |

## Limitations

- Clusters describe behavioral similarity, not causal drivers of spending.
- Annual income appears to be measured in full currency units rather than `k$`; interpretation uses relative income bands.
- Agglomerative clustering and DBSCAN are evaluated on deterministic samples to keep runtime practical for this dataset size.
- Segment names are business-friendly summaries of cluster averages and should be validated with domain context before campaign launch.

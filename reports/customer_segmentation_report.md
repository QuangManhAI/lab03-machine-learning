# Customer Segmentation Report

## Dataset

- Raw file: `/Users/quangmanh/Project/lab03/data/raw/Shopping Mall Customer Segmentation Data .csv`
- The dataset contains customer ID, gender, age, annual income, and spending score.
- Customer ID is preserved for traceability but excluded from clustering.
- Gender is compared through a one-hot encoded experiment rather than ordinal encoding.

## Selected Model

- Algorithm: `kmeans`
- Feature set: `income_spending_only`
- Scaler: `standard`
- Parameters: `n_clusters=4`
- Number of clusters: `4`
- Silhouette: `0.4080`
- Calinski-Harabasz: `15050.68`
- Davies-Bouldin: `0.7687`

The selected solution balances quantitative quality with interpretability. The final choice prefers 3 to 7 clusters, strong silhouette performance, a lower Davies-Bouldin score, and segments that are readable in income/spending plots.

## Segment Profiles

|   cluster |   count |   age_mean |   age_median |   annual_income_mean |   annual_income_median |   spending_score_mean |   spending_score_median |   percentage |   gender_share_Female |   gender_share_Male | segment_name               | recommended_business_action                                                  |
|----------:|--------:|-----------:|-------------:|---------------------:|-----------------------:|----------------------:|------------------------:|-------------:|----------------------:|--------------------:|:---------------------------|:-----------------------------------------------------------------------------|
|         0 |    3647 |    54.1829 |           54 |             156283   |               156431   |               25.5755 |                      25 |       0.2419 |                0.5062 |              0.4938 | High income, low spending  | Use personalized recommendations and targeted incentives to lift conversion. |
|         1 |    3783 |    53.9688 |           54 |              64904.2 |                65131   |               75.511  |                      76 |       0.2509 |                0.4951 |              0.5049 | Low income, high spending  | Offer bundles, discounts, and value-focused retention campaigns.             |
|         2 |    3864 |    54.2513 |           54 |              65171.1 |                64459.5 |               25.7896 |                      26 |       0.2563 |                0.4902 |              0.5098 | Low income, low spending   | Use low-cost engagement and avoid expensive acquisition-style campaigns.     |
|         3 |    3785 |    54.3617 |           55 |             155216   |               155004   |               75.1091 |                      75 |       0.251  |                0.4943 |              0.5057 | High income, high spending | Prioritize loyalty perks, premium offers, and early access campaigns.        |

## Model Comparison Snapshot

| algorithm   | feature_set          | scaler   | params                   |   n_clusters |   noise_rate |   silhouette |   calinski_harabasz |   davies_bouldin | selected   |
|:------------|:---------------------|:---------|:-------------------------|-------------:|-------------:|-------------:|--------------------:|-----------------:|:-----------|
| kmeans      | income_spending_only | standard | n_clusters=4             |            4 |     0        |     0.407971 |           15050.7   |         0.768664 | True       |
| dbscan      | behavior_plus_gender | standard | eps=0.25, min_samples=10 |            2 |     0.994    |     0.887774 |             733.039 |         0.15038  | False      |
| kmeans      | behavior_plus_gender | minmax   | n_clusters=2             |            2 |     0        |     0.57873  |           29617.2   |         0.685333 | False      |
| dbscan      | behavior_plus_gender | standard | eps=0.5, min_samples=35  |            6 |     0.932571 |     0.551442 |             593.091 |         0.720362 | False      |
| kmeans      | income_spending_only | minmax   | n_clusters=4             |            4 |     0        |     0.407974 |           15050.9   |         0.768782 | False      |
| kmeans      | income_spending_only | robust   | n_clusters=4             |            4 |     0        |     0.407801 |           15045     |         0.771876 | False      |
| kmeans      | age_spending_only    | minmax   | n_clusters=4             |            4 |     0        |     0.406911 |           14911.2   |         0.773988 | False      |
| kmeans      | age_spending_only    | robust   | n_clusters=4             |            4 |     0        |     0.40677  |           14911.7   |         0.770727 | False      |
| kmeans      | age_spending_only    | standard | n_clusters=4             |            4 |     0        |     0.406769 |           14911.7   |         0.770603 | False      |
| kmeans      | behavior_plus_gender | minmax   | n_clusters=3             |            3 |     0        |     0.403335 |           18103.8   |         1.29108  | False      |
| kmeans      | age_spending_only    | standard | n_clusters=5             |            5 |     0        |     0.391629 |           14012.8   |         0.83804  | False      |
| kmeans      | age_spending_only    | robust   | n_clusters=5             |            5 |     0        |     0.391538 |           14012.9   |         0.837918 | False      |

## Limitations

- Clusters describe behavioral similarity, not causal drivers of spending.
- Annual income appears to be measured in full currency units rather than `k$`; interpretation uses relative income bands.
- Agglomerative clustering and DBSCAN are evaluated on deterministic samples to keep runtime practical for this dataset size.
- Segment names are business-friendly summaries of cluster averages and should be validated with domain context before campaign launch.

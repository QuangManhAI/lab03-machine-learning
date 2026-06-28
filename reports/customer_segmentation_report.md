# Customer Segmentation Report

## Dataset

- Raw file: `C:\Users\USER\lab03-machine-learning\data\raw\datav2.csv`
- The dataset contains customer ID, gender, age, annual income, and spending score.
- Customer ID is preserved for traceability but excluded from clustering.
- Gender is compared through a one-hot encoded experiment rather than ordinal encoding.

## Selected Model

- Algorithm: `kmeans`
- Feature set: `income_spending_only`
- Scaler: `robust`
- Parameters: `n_clusters=5`
- Number of clusters: `5`
- Silhouette: `0.4702`

The selected solution balances quantitative quality with interpretability. The final choice prefers 3 to 7 clusters, strong silhouette performance, and segments that are readable in income/spending plots.

## Segment Profiles

|   cluster |   count |   age_mean |   age_median |   annual_income_mean |   annual_income_median |   spending_score_mean |   spending_score_median |   percentage |   gender_share_Female |   gender_share_Male | segment_name                       | recommended_business_action                                                            |
|----------:|--------:|-----------:|-------------:|---------------------:|-----------------------:|----------------------:|------------------------:|-------------:|----------------------:|--------------------:|:-----------------------------------|:---------------------------------------------------------------------------------------|
|         0 |     497 |    48.9799 |           47 |              91230.8 |                  89100 |               16.6338 |                      17 |       0.2259 |                0.5131 |              0.4869 | High income, low spending          | Use personalized recommendations and targeted incentives to lift conversion.           |
|         1 |     258 |    42.0039 |           42 |              25957   |                  25000 |               22.1124 |                      23 |       0.1173 |                0.562  |              0.438  | Low income, low spending           | Use low-cost engagement and avoid expensive acquisition-style campaigns.               |
|         2 |     712 |    38.0098 |           37 |              53151.7 |                  52100 |               47.6517 |                      48 |       0.3236 |                0.5379 |              0.4621 | Moderate income, moderate spending | Use broad retention campaigns, seasonal offers, and monitor movement between segments. |
|         3 |     242 |    33      |           33 |              83886.4 |                  83000 |               80.7231 |                      81 |       0.11   |                0.5496 |              0.4504 | High income, high spending         | Prioritize loyalty perks, premium offers, and early access campaigns.                  |
|         4 |     491 |    27.9572 |           27 |              27151.5 |                  27000 |               70.4053 |                      70 |       0.2232 |                0.558  |              0.442  | Low income, high spending          | Offer bundles, discounts, and value-focused retention campaigns.                       |

## Model Comparison Snapshot

| algorithm   | feature_set          | scaler   | params         |   n_clusters |   noise_rate |   silhouette | selected   |
|:------------|:---------------------|:---------|:---------------|-------------:|-------------:|-------------:|:-----------|
| kmeans      | income_spending_only | robust   | n_clusters=5   |            5 |            0 |     0.470209 | True       |
| kmeans      | behavior_plus_gender | minmax   | n_clusters=2   |            2 |            0 |     0.68097  | False      |
| kmeans      | behavior_plus_gender | minmax   | n_clusters=3   |            3 |            0 |     0.515225 | False      |
| kmeans      | age_spending_only    | minmax   | n_clusters=2   |            2 |            0 |     0.473303 | False      |
| kmeans      | income_spending_only | standard | n_clusters=5   |            5 |            0 |     0.468631 | False      |
| kmeans      | income_spending_only | minmax   | n_clusters=5   |            5 |            0 |     0.468489 | False      |
| kmeans      | age_spending_only    | robust   | n_clusters=2   |            2 |            0 |     0.462454 | False      |
| kmeans      | age_spending_only    | standard | n_clusters=2   |            2 |            0 |     0.461165 | False      |
| gmm         | income_spending_only | standard | n_components=5 |            5 |            0 |     0.451776 | False      |
| gmm         | income_spending_only | standard | n_components=4 |            4 |            0 |     0.434551 | False      |
| gmm         | income_spending_only | standard | n_components=6 |            6 |            0 |     0.398387 | False      |
| gmm         | income_spending_only | standard | n_components=2 |            2 |            0 |     0.392542 | False      |

## Limitations

- Clusters describe behavioral similarity, not causal drivers of spending.
- Annual income appears to be measured in full currency units rather than `k$`; interpretation uses relative income bands.
- Segment names are business-friendly summaries of cluster averages and should be validated with domain context before campaign launch.

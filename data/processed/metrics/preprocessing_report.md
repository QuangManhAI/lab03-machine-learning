# Preprocessing Report

- Raw shape: `2200` rows x `5` columns
- Clean shape: `2200` rows x `5` columns
- Raw data was left unchanged under `data/raw/`.
- Customer IDs were preserved but excluded from clustering features.
- Gender was kept categorical and only one-hot encoded for the gender experiment.

## Missing Values

| column         |   missing_count |   missing_rate |
|:---------------|----------------:|---------------:|
| age            |               0 |              0 |
| annual_income  |               0 |              0 |
| customer_id    |               0 |              0 |
| gender         |               0 |              0 |
| spending_score |               0 |              0 |

## Duplicate Checks

| check                  |   count |   rate |
|:-----------------------|--------:|-------:|
| duplicate_rows         |       0 |      0 |
| duplicate_customer_ids |       0 |      0 |

## Invalid Value Checks

| check                  |   count |   rate |
|:-----------------------|--------:|-------:|
| invalid_age            |       0 |      0 |
| invalid_annual_income  |       0 |      0 |
| invalid_spending_score |       0 |      0 |

## Outlier Summary

| column         |    q1 |    q3 |   iqr |   iqr_lower |   iqr_upper |   iqr_outliers |   zscore_outliers |
|:---------------|------:|------:|------:|------------:|------------:|---------------:|------------------:|
| age            |    30 |    44 |    14 |         9   |        65   |             88 |                29 |
| annual_income  | 31875 | 79525 | 47650 |    -39600   |    151000   |              0 |                 1 |
| spending_score |    26 |    65 |    39 |       -32.5 |       123.5 |              0 |                 0 |

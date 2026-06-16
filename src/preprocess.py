from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler


REQUIRED_COLUMNS = ["customer_id", "gender", "age", "annual_income", "spending_score"]
NUMERIC_COLUMNS = ["age", "annual_income", "spending_score"]


def load_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def _safe_name(column: str) -> str:
    name = column.strip().lower()
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    aliases = {
        "customerid": "customer_id",
        "customer_id": "customer_id",
        "id": "customer_id",
        "genre": "gender",
        "sex": "gender",
        "annual_income": "annual_income",
        "annual_income_k": "annual_income",
        "income": "annual_income",
        "spending_score": "spending_score",
        "spending_score_1_100": "spending_score",
        "score": "spending_score",
    }
    return aliases.get(name, name)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [_safe_name(column) for column in renamed.columns]
    return renamed


def validate_required_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
        }
    )
    return summary.sort_values(["missing_count", "column"], ascending=[False, True])


def duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_rows = int(df.duplicated().sum())
    duplicate_customer_ids = int(df["customer_id"].duplicated().sum()) if "customer_id" in df else 0
    return pd.DataFrame(
        [
            {"check": "duplicate_rows", "count": duplicate_rows, "rate": duplicate_rows / len(df)},
            {
                "check": "duplicate_customer_ids",
                "count": duplicate_customer_ids,
                "rate": duplicate_customer_ids / len(df),
            },
        ]
    )


def invalid_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "invalid_age": (df["age"] < 0) | (df["age"] > 120),
        "invalid_annual_income": df["annual_income"] < 0,
        "invalid_spending_score": (df["spending_score"] < 1) | (df["spending_score"] > 100),
    }
    return pd.DataFrame(
        [{"check": name, "count": int(mask.sum()), "rate": float(mask.mean())} for name, mask in checks.items()]
    )


def outlier_summary(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = df[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (df[column] < lower) | (df[column] > upper)
        std = series.std(ddof=0)
        z_mask = pd.Series(False, index=df.index) if std == 0 else ((df[column] - series.mean()).abs() / std) > 3
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "iqr_lower": lower,
                "iqr_upper": upper,
                "iqr_outliers": int(iqr_mask.sum()),
                "zscore_outliers": int(z_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = normalize_column_names(df)
    validate_required_columns(clean)
    clean = clean.drop_duplicates().copy()
    clean["customer_id"] = clean["customer_id"].astype(str).str.strip()
    clean["gender"] = clean["gender"].astype(str).str.strip().str.title()
    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=REQUIRED_COLUMNS)
    clean = clean[(clean["age"].between(0, 120)) & (clean["annual_income"] >= 0)]
    clean = clean[clean["spending_score"].between(1, 100)]
    clean["age"] = clean["age"].astype(int)
    clean["annual_income"] = clean["annual_income"].astype(float)
    clean["spending_score"] = clean["spending_score"].astype(float)
    return clean.reset_index(drop=True)


def make_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[NUMERIC_COLUMNS].describe().T.reset_index().rename(columns={"index": "column"})
    numeric["missing_count"] = numeric["column"].map(df[NUMERIC_COLUMNS].isna().sum())
    return numeric


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "behavior_numeric",
    scaler_name: str = "standard",
) -> tuple[np.ndarray, list[str], ColumnTransformer]:
    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if scaler_name not in scaler_map:
        raise ValueError(f"Unknown scaler: {scaler_name}")

    numeric_features = list(NUMERIC_COLUMNS)
    categorical_features: list[str] = []
    if feature_set == "behavior_plus_gender":
        categorical_features = ["gender"]
    elif feature_set == "income_spending_only":
        numeric_features = ["annual_income", "spending_score"]
    elif feature_set == "age_spending_only":
        numeric_features = ["age", "spending_score"]
    elif feature_set != "behavior_numeric":
        raise ValueError(f"Unknown feature set: {feature_set}")

    transformers = [("numeric", scaler_map[scaler_name], numeric_features)]
    if categorical_features:
        transformers.append(("gender", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features))

    transformer = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    matrix = transformer.fit_transform(df)
    feature_names = list(transformer.get_feature_names_out())
    return np.asarray(matrix, dtype=float), feature_names, transformer


def transform_feature_matrix(df: pd.DataFrame, transformer: ColumnTransformer) -> np.ndarray:
    return np.asarray(transformer.transform(df), dtype=float)


def inverse_numeric_centers(
    centers: np.ndarray,
    transformer: ColumnTransformer,
    numeric_features: list[str],
) -> pd.DataFrame:
    scaler = transformer.named_transformers_["numeric"]
    numeric_centers = centers[:, : len(numeric_features)]
    original = scaler.inverse_transform(numeric_centers)
    return pd.DataFrame(original, columns=numeric_features)


def write_preprocessing_report(
    path: Path,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    duplicate_checks: pd.DataFrame,
    invalid_checks: pd.DataFrame,
    outliers: pd.DataFrame,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        "# Preprocessing Report",
        "",
        f"- Raw shape: `{raw_df.shape[0]}` rows x `{raw_df.shape[1]}` columns",
        f"- Clean shape: `{clean_df.shape[0]}` rows x `{clean_df.shape[1]}` columns",
        "- Raw data was left unchanged under `data/raw/`.",
        "- Customer IDs were preserved but excluded from clustering features.",
        "- Gender was kept categorical and only one-hot encoded for the gender experiment.",
        "",
        "## Missing Values",
        "",
        missing_summary.to_markdown(index=False),
        "",
        "## Duplicate Checks",
        "",
        duplicate_checks.to_markdown(index=False),
        "",
        "## Invalid Value Checks",
        "",
        invalid_checks.to_markdown(index=False),
        "",
        "## Outlier Summary",
        "",
        outliers.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return path

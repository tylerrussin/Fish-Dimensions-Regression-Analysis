"""Модуль автоматических проверок качества данных для EDA FishGrow."""
import numpy as np
import pandas as pd


def basic_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Форма, типы, пропуски, уникальность."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=False),
    })


def check_duplicates(df: pd.DataFrame) -> dict:
    """Дубликаты строк и потенциальные ID."""
    return {
        "rows": len(df),
        "full_duplicates": int(df.duplicated().sum()),
        "duplicated_ids": [
            c for c in df.columns if df[c].nunique() == len(df)
        ],
    }


def check_infinities(df: pd.DataFrame) -> pd.Series:
    """Бесконечности в числовых полях."""
    num = df.select_dtypes(include=[np.number])
    return np.isinf(num).sum()


def check_positive(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Отрицательные и нулевые значения там, где физически должно быть > 0."""
    return (df[cols] <= 0).sum()


def check_length_order(df: pd.DataFrame) -> dict:
    """Согласованность длин: обычно Length1 <= Length2 <= Length3."""
    return {
        "L1_gt_L2": int((df["Length1"] > df["Length2"]).sum()),
        "L2_gt_L3": int((df["Length2"] > df["Length3"]).sum()),
        "L1_gt_L3": int((df["Length1"] > df["Length3"]).sum()),
    }


def check_volume_consistency(df: pd.DataFrame) -> pd.Series:
    """Объём должен быть > 0 и согласован по порядку величины с массой."""
    v = df["Length3"] * df["Height"] * df["Width"]
    return pd.Series({
        "V_le_0": int((v <= 0).sum()),
        "corr_V_Weight": round(v.corr(df["Weight"]), 3),
    })


def rare_categories(df: pd.DataFrame, col: str, threshold: float = 0.05) -> pd.Series:
    """Редкие категории (< threshold от всех наблюдений)."""
    freq = df[col].value_counts(normalize=True)
    return freq[freq < threshold]


def almost_duplicate_rows(df: pd.DataFrame, tol: float = 0.05) -> int:
    """Почти одинаковые строки по числовым признакам."""
    num = df.select_dtypes(include=[np.number]).copy()
    # нормализация, чтобы признаки были сопоставимы
    num = (num - num.mean()) / num.std(ddof=0)
    num = num.fillna(0)
    vals = num.round(2).astype(str).agg("|".join, axis=1)
    return int(vals.duplicated().sum())
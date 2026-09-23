"""Переиспользуемые проверки качества данных для Fish.csv.

Модуль используется из notebooks/01-eda.ipynb и tests/.
Не зависит от pandas-display и работает как библиотека.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "Species", "Weight",
    "Length1", "Length2", "Length3",
    "Height", "Width",
]

NUMERIC_COLUMNS = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
GEOMETRIC_COLUMNS = ["Length1", "Length2", "Length3", "Height", "Width"]

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "data" / "Fish.csv"
)


def load(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Читает датасет, не меняя его."""
    return pd.read_csv(path)


def check_shape_dtypes(df: pd.DataFrame) -> dict:
    """Форма, список столбцов, типы."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "columns_match_expected": list(df.columns) == EXPECTED_COLUMNS,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    """Полные дубликаты строк."""
    dup_mask = df.duplicated(keep=False)
    return {
        "n_full_duplicates": int(df.duplicated().sum()),
        "duplicate_rows": df[dup_mask].index.tolist(),
    }


def check_missing(df: pd.DataFrame) -> dict:
    """Пропуски по столбцам."""
    return {
        "missing_per_column": df.isna().sum().to_dict(),
        "total_missing": int(df.isna().sum().sum()),
    }


def check_infinities(df: pd.DataFrame) -> dict:
    """Бесконечности в числовых столбцах."""
    num = df.select_dtypes(include="number")
    inf_mask = num.apply(lambda s: s.map(lambda x: isinstance(x, float) and math.isinf(x)))
    return {
        "inf_per_column": inf_mask.sum().to_dict(),
        "total_inf": int(inf_mask.sum().sum()),
    }


def check_geometry_consistency(df: pd.DataFrame) -> dict:
    """Согласованность геометрических измерений и положительность."""
    report: dict = {}

    # все геометрические > 0
    for col in GEOMETRIC_COLUMNS:
        report[f"{col}_positive"] = bool((df[col] > 0).all())
        report[f"{col}_min"] = float(df[col].min())
        report[f"{col}_max"] = float(df[col].max())

    # Weight > 0 (после очистки)
    report["Weight_positive"] = bool((df["Weight"] > 0).all())

    # порядок длин
    report["Length1_le_Length2"] = bool((df["Length1"] <= df["Length2"] + 1e-6).all())
    report["Length2_le_Length3"] = bool((df["Length2"] <= df["Length3"] + 1e-6).all())

    # высота/ширина меньше длины
    report["Height_lt_Length3"] = bool((df["Height"] < df["Length3"]).all())
    report["Width_lt_Length3"] = bool((df["Width"] < df["Length3"]).all())

    return report


def check_rare_categories(df: pd.DataFrame, column: str = "Species", min_count: int = 5) -> dict:
    """Редкие категории и дисбаланс."""
    counts = df[column].value_counts()
    return {
        "counts": counts.to_dict(),
        "n_categories": int(counts.shape[0]),
        "rare_categories": counts[counts < min_count].to_dict(),
        "imbalance_ratio": float(counts.max() / counts.min()) if counts.min() > 0 else float("inf"),
    }


def check_near_duplicates(df: pd.DataFrame, tol: float = 1e-6, ignore: list[str] | None = None) -> dict:
    """Почти одинаковые строки по числовым столбцам.

    Строки считаются почти одинаковыми, если все числовые столбцы
    совпадают с точностью tol, а Species совпадает.
    """
    ignore = ignore or []
    cols = [c for c in df.columns if c not in ignore]
    rounded = df[cols].copy()
    for c in NUMERIC_COLUMNS:
        if c in rounded:
            rounded[c] = rounded[c].round(int(-math.log10(tol)))
    dup_mask = rounded.duplicated(keep=False)
    return {
        "n_near_duplicates": int(rounded.duplicated().sum()),
        "near_duplicate_rows": df[dup_mask].index.tolist(),
    }


def check_unique_like_columns(df: pd.DataFrame, threshold: float = 0.99) -> dict:
    """Столбцы, значения которых почти уникальны.

    Уникальные поля могут быть идентификаторами или источниками утечки,
    если они кодируют информацию о целевом значении.
    """
    n = len(df)
    result = {}
    for col in df.columns:
        uniqueness = df[col].nunique(dropna=False) / n
        result[col] = {
            "n_unique": int(df[col].nunique(dropna=False)),
            "uniqueness_ratio": round(float(uniqueness), 4),
            "looks_like_identifier": bool(uniqueness >= threshold),
        }
    return result


def run_all(df: pd.DataFrame) -> dict:
    """Прогоняет все проверки и возвращает сводный отчёт."""
    return {
        "shape_dtypes": check_shape_dtypes(df),
        "duplicates": check_duplicates(df),
        "missing": check_missing(df),
        "infinities": check_infinities(df),
        "geometry": check_geometry_consistency(df),
        "rare_categories": check_rare_categories(df, "Species"),
        "near_duplicates": check_near_duplicates(df),
        "unique_like": check_unique_like_columns(df),
    }


if __name__ == "__main__":
    import json
    df = load()
    report = run_all(df)
    print(json.dumps(report, indent=2, ensure_ascii=False))

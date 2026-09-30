"""
Модуль автоматических проверок качества данных для FishGrow (лаб. №1).

Использование из блокнота:

    import sys
    from pathlib import Path
    sys.path.append(str(Path("..") / "src"))
    import importlib
    import data_checks as dc
    importlib.reload(dc)

    report = dc.run_all_checks(df)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Форма, типы, дубликаты
# ---------------------------------------------------------------------------

def check_shape(df, expected_rows=None, expected_cols=None):
    """Размер таблицы: строки, столбцы, соответствие ожиданиям."""
    rows, cols = df.shape
    return {
        "check": "shape",
        "rows": rows, "cols": cols,
        "rows_ok": expected_rows is None or rows == expected_rows,
        "cols_ok": expected_cols is None or cols == expected_cols,
    }


def check_dtypes(df, expected=None):
    """Типы данных по столбцам; опционально — сравнение с ожидаемыми."""
    result = pd.DataFrame({"dtype": df.dtypes.astype(str)})
    if expected:
        result["expected"] = pd.Series(expected)
        result["ok"] = result["dtype"] == result["expected"]
    return result


def check_duplicates(df, subset=None):
    """Полные дубликаты строк (или по заданному набору столбцов)."""
    mask = df.duplicated(keep=False, subset=subset)
    return {
        "check": "duplicates",
        "subset": subset,
        "n_duplicated_rows": int(mask.sum()),
        "duplicated_indices": df.index[mask].tolist(),
    }


# ---------------------------------------------------------------------------
# 2. Пропуски, бесконечности, невозможные значения
# ---------------------------------------------------------------------------

def check_missing(df):
    """Пропуски и их доля по каждому столбцу."""
    return pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })


def check_infinities(df):
    """Бесконечности (положительные и отрицательные) в числовых столбцах."""
    num = df.select_dtypes("number")
    return pd.DataFrame({
        "pos_inf": np.isposinf(num).sum(),
        "neg_inf": np.isneginf(num).sum(),
    })


def check_positive(df, cols):
    """Все значения в указанных столбцах должны быть строго > 0."""
    result = {}
    for c in cols:
        if c not in df.columns:
            continue
        mask = df[c] <= 0
        result[c] = {
            "n_violations": int(mask.sum()),
            "indices": df.index[mask].tolist(),
            "min_value": float(df[c].min()),
        }
    return result


# ---------------------------------------------------------------------------
# 3. Согласованность геометрических измерений
# ---------------------------------------------------------------------------

def check_length_order(df, cols=("Length1", "Length2", "Length3")):
    """Length1 <= Length2 <= Length3 — три способа измерения длины одной рыбы."""
    a, b, c = cols
    if not all(x in df.columns for x in cols):
        return {"check": "length_order", "error": "columns not found"}
    bad = (df[a] > df[b]) | (df[b] > df[c])
    return {
        "check": "length_order",
        "rule": f"{a} <= {b} <= {c}",
        "n_violations": int(bad.sum()),
        "indices": df.index[bad].tolist(),
    }


def check_volume_consistency(df):
    """Проверка согласованности Height и Width — грубые отношения."""
    if not {"Height", "Width"}.issubset(df.columns):
        return {"check": "volume_consistency", "error": "columns not found"}
    ratio = (df["Height"] / df["Width"]).round(3)
    return {
        "check": "volume_consistency",
        "height_width_ratio_min": float(ratio.min()),
        "height_width_ratio_max": float(ratio.max()),
        "height_width_ratio_median": float(ratio.median()),
    }


# ---------------------------------------------------------------------------
# 4. Категории, редкие виды, дисбаланс
# ---------------------------------------------------------------------------

def check_categories(df, col):
    """Численность и доля каждой категории."""
    vc = df[col].value_counts(dropna=False)
    pct = (vc / len(df) * 100).round(2)
    return pd.DataFrame({"count": vc, "pct": pct})


def check_rare_categories(df, col, threshold=5.0):
    """Категории с долей меньше порога threshold (в %)."""
    vc = df[col].value_counts()
    pct = vc / len(df) * 100
    rare = pct[pct < threshold]
    return {
        "check": "rare_categories",
        "column": col,
        "threshold_pct": threshold,
        "rare": rare.round(2).to_dict(),
        "n_rare": int(len(rare)),
    }


def check_imbalance(df, col):
    """Степень дисбаланса: во сколько раз самый частый класс больше самого редкого."""
    vc = df[col].value_counts()
    return {
        "check": "imbalance",
        "column": col,
        "n_classes": int(vc.shape[0]),
        "majority": vc.index[0],
        "majority_count": int(vc.iloc[0]),
        "minority": vc.index[-1],
        "minority_count": int(vc.iloc[-1]),
        "ratio_majority_minority": float(vc.iloc[0] / vc.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# 5. Почти одинаковые строки
# ---------------------------------------------------------------------------

def check_near_duplicates(df, numeric_cols, atol=1e-6):
    """Пары строк, у которых все числовые признаки совпадают в пределах atol."""
    num = df[numeric_cols].to_numpy(dtype=float)
    n = len(num)
    pairs = []
    for i, j in combinations(range(n), 2):
        if np.allclose(num[i], num[j], atol=atol):
            pairs.append((i, j))
    return {
        "check": "near_duplicates",
        "n_pairs": len(pairs),
        "pairs": pairs[:20],  # первые 20 пар для обозримости
    }


# ---------------------------------------------------------------------------
# 6. Уникальные поля — кандидаты в идентификаторы / источники утечки
# ---------------------------------------------------------------------------

def check_id_like(df, max_unique_ratio=0.95):
    """Столбцы, где уникальных значений почти столько же, сколько строк."""
    result = {}
    n = len(df)
    for c in df.columns:
        n_unique = df[c].nunique(dropna=False)
        ratio = n_unique / n if n else 0.0
        if ratio >= max_unique_ratio:
            result[c] = {"n_unique": n_unique, "ratio": round(ratio, 3)}
    return {
        "check": "id_like",
        "threshold": max_unique_ratio,
        "candidates": result,
    }


# ---------------------------------------------------------------------------
# Полный отчёт
# ---------------------------------------------------------------------------

def run_all_checks(df):
    """Запускает все проверки, возвращает словарь с результатами."""
    numeric_cols = df.select_dtypes("number").columns.tolist()
    positive_cols = [c for c in
                     ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
                     if c in df.columns]

    return {
        "shape": check_shape(df, expected_rows=159, expected_cols=7),
        "dtypes": check_dtypes(df),
        "duplicates": check_duplicates(df),
        "missing": check_missing(df),
        "infinities": check_infinities(df),
        "positivity": check_positive(df, positive_cols),
        "length_order": check_length_order(df),
        "volume_consistency": check_volume_consistency(df),
        "species_dist": check_categories(df, "Species") if "Species" in df.columns else None,
        "rare_categories": check_rare_categories(df, "Species", threshold=10.0) if "Species" in df.columns else None,
        "imbalance": check_imbalance(df, "Species") if "Species" in df.columns else None,
        "near_duplicates": check_near_duplicates(df, numeric_cols),
        "id_like": check_id_like(df),
    }

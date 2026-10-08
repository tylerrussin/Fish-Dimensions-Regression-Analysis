"""
Переиспользуемые проверки качества данных для датасета Fish.csv
(и любых похожих табличных датасетов).

Все функции:
- принимают pandas.DataFrame и (при необходимости) параметры;
- НИЧЕГО не печатают и не меняют входные данные;
- возвращают либо DataFrame-отчёт, либо словарь, либо bool.

Пример использования в ноутбуке:

    from src.data_checks import (
        check_shape, check_dtypes, check_duplicates, check_id_columns,
        check_missing_and_infinite, check_impossible_values,
        check_geometric_consistency,
        check_rare_categories, check_class_balance,
        check_near_duplicates, check_unique_leakage_columns,
    )

    report = run_all_checks(df, target="Weight", group="Species")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

# ---------------------------------------------------------------------------
# Константы предметной области Fish.csv
# ---------------------------------------------------------------------------

NUMERIC_COLS = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
TARGET_COL = "Weight"
GROUP_COL = "Species"
KNOWN_SPECIES = ["Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"]


# ---------------------------------------------------------------------------
# 1. Форма, типы, дубликаты, идентификаторы
# ---------------------------------------------------------------------------

def check_shape(df: pd.DataFrame) -> dict:
    """Возвращает форму таблицы и число ячеек."""
    return {"rows": df.shape[0], "columns": df.shape[1], "cells": df.size}


def check_dtypes(
    df: pd.DataFrame,
    expected: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Отчёт по типам данных.
    expected — необязательный словарь {колонка: 'numeric'|'category'|'string'}.
    """
    expected = expected or {
        "Species": "str",
        **{c: "numeric" for c in NUMERIC_COLS},
    }
    rows = []
    for col, want in expected.items():
        if col not in df.columns:
            rows.append({"column": col, "expected": want, "actual": None,
                         "ok": False, "note": "колонка отсутствует"})
            continue
        actual = df[col].dtype
        if want == "numeric":
            ok = pd.api.types.is_numeric_dtype(actual)
        elif want == "category":
            ok = isinstance(actual, pd.CategoricalDtype) or actual == object
        else:
            ok = True
        rows.append({"column": col, "expected": want, "actual": str(actual),
                     "ok": ok, "note": ""})
    return pd.DataFrame(rows)


def check_duplicates(df: pd.DataFrame) -> dict:
    """Полные дубликаты строк (все колонки совпадают)."""
    dup_mask = df.duplicated(keep=False)
    return {
        "n_duplicated_rows": int(df.duplicated().sum()),
        "n_rows_in_dup_groups": int(dup_mask.sum()),
        "duplicate_examples": df[dup_mask].head(20),
    }


def check_id_columns(
    df: pd.DataFrame,
    candidate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Проверяет колонки-кандидаты на роль идентификатора:
    уникальность, отсутствие пропусков, непостоянность.
    """
    candidate_cols = candidate_cols or df.columns.tolist()
    rows = []
    n = len(df)
    for col in candidate_cols:
        s = df[col]
        n_unique = s.nunique(dropna=False)
        is_unique = n_unique == n
        has_na = s.isna().any()
        is_constant = n_unique <= 1
        rows.append({
            "column": col,
            "n_unique": int(n_unique),
            "unique_ratio": round(n_unique / n, 4) if n else 0.0,
            "is_unique": is_unique,
            "has_missing": bool(has_na),
            "is_constant": bool(is_constant),
            "candidate_id": is_unique and not has_na and not is_constant,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Пропуски, бесконечности, невозможные значения
# ---------------------------------------------------------------------------

def check_missing_and_infinite(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Пропуски и ±inf по всем колонкам (для числовых — отдельно inf)."""
    numeric_cols = numeric_cols or NUMERIC_COLS
    rows = []
    for col in df.columns:
        s = df[col]
        n_na = int(s.isna().sum())
        n_inf = 0
        if col in numeric_cols and pd.api.types.is_numeric_dtype(s):
            n_inf = int(np.isinf(s).sum())
        rows.append({
            "column": col,
            "n_missing": n_na,
            "pct_missing": round(100 * n_na / len(df), 2) if len(df) else 0.0,
            "n_infinite": n_inf,
            "ok": n_na == 0 and n_inf == 0,
        })
    return pd.DataFrame(rows)


def check_impossible_values(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Невозможные значения для Fish.csv:
    - неположительные размеры и вес (<= 0);
    - вес ровно 0 (физически невозможно, замаскированный пропуск);
    - отрицательные значения.
    """
    numeric_cols = numeric_cols or NUMERIC_COLS
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "column": col,
            "n_le_zero": int((s <= 0).sum()),
            "n_lt_zero": int((s < 0).sum()),
            "n_eq_zero": int((s == 0).sum()),
            "min": s.min(),
            "max": s.max(),
            "ok": bool((s > 0).all()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Согласованность геометрических измерений
# ---------------------------------------------------------------------------

def check_geometric_consistency(df: pd.DataFrame) -> dict:
    """
    Анатомические правила для Fish.csv:
      Length1 < Length2 < Length3,
      Height  < Length3,
      Width   < Height. УБРАЛ НАХУЙ
    Возвращает словарь с масками нарушений и количеством.
    """
    required = ["Length1", "Length2", "Length3", "Height", "Width"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"ok": False, "error": f"нет колонок: {missing}"}

    rules = {
        "Length1_lt_Length2": df["Length1"] < df["Length2"],
        "Length2_lt_Length3": df["Length2"] < df["Length3"],
        "Height_lt_Length3":  df["Height"]  < df["Length3"],
    }
    violations = {name: (~mask) for name, mask in rules.items()}
    summary = {name: int(v.sum()) for name, v in violations.items()}
    bad_rows_mask = np.logical_or.reduce([v.values for v in violations.values()])
    return {
        "ok": all(v == 0 for v in summary.values()),
        "violations_count": summary,
        "bad_rows": df[bad_rows_mask].copy(),
    }


# ---------------------------------------------------------------------------
# 4. Редкие категории и дисбаланс
# ---------------------------------------------------------------------------

def check_rare_categories(
    df: pd.DataFrame,
    col: str,
    min_count: int = 5,
    min_freq: float = 0.01,
) -> pd.DataFrame:
    """Отчёт по частотам категорий; помечает редкие."""
    vc = df[col].value_counts(dropna=False)
    freq = df[col].value_counts(normalize=True, dropna=False)
    out = pd.DataFrame({"count": vc, "freq": freq.round(4)})
    out["rare_by_count"] = out["count"] < min_count
    out["rare_by_freq"] = out["freq"] < min_freq
    out["is_rare"] = out["rare_by_count"] | out["rare_by_freq"]
    return out.sort_values("count", ascending=False)


def check_class_balance(
    df: pd.DataFrame,
    col: str,
) -> dict:
    """Метрики дисбаланса классов: доля мажоритарного, соотношение max/min."""
    vc = df[col].value_counts()
    if vc.empty:
        return {"ok": False, "error": "нет данных"}
    total = vc.sum()
    return {
        "n_classes": int(vc.shape[0]),
        "majority_class": vc.idxmax(),
        "majority_share": round(vc.max() / total, 4),
        "minority_class": vc.idxmin(),
        "minority_share": round(vc.min() / total, 4),
        "imbalance_ratio_max_min": round(vc.max() / vc.min(), 2) if vc.min() > 0 else np.inf,
        "counts": vc.to_dict(),
    }


# ---------------------------------------------------------------------------
# 5. Почти одинаковые строки
# ---------------------------------------------------------------------------

def check_near_duplicates(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> pd.DataFrame:
    """
    Ищет пары строк, у которых ВСЕ числовые колонки совпадают
    с точностью rtol/atol. Возвращает таблицу пар.
    """
    numeric_cols = numeric_cols or [c for c in NUMERIC_COLS if c in df.columns]
    idx = df.index.to_numpy()
    pairs = []
    sub = df[numeric_cols]
    for i, j in combinations(range(len(df)), 2):
        a, b = sub.iloc[i], sub.iloc[j]
        if np.allclose(a.to_numpy(dtype=float),
                       b.to_numpy(dtype=float),
                       rtol=rtol, atol=atol, equal_nan=True):
            pairs.append({
                "idx_a": idx[i], "idx_b": idx[j],
                **{f"{c}_a": a[c] for c in numeric_cols},
                **{f"{c}_b": b[c] for c in numeric_cols},
            })
    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# 6. Уникальные поля — потенциальные ID или утечки
# ---------------------------------------------------------------------------

def check_unique_leakage_columns(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Помечает колонки, которые:
    - почти уникальны (уникальность >= threshold) → потенциальный ID,
    - детерминированно связаны с целевой (|corr| >= threshold) → утечка.
    """
    rows = []
    n = len(df)
    target_num = pd.to_numeric(df[target], errors="coerce") if target in df else None
    for col in df.columns:
        s = df[col]
        n_unique = s.nunique(dropna=False)
        unique_ratio = n_unique / n if n else 0.0
        corr = np.nan
        if (col != target and target_num is not None
                and pd.api.types.is_numeric_dtype(s)):
            corr = s.corr(target_num)
        rows.append({
            "column": col,
            "n_unique": int(n_unique),
            "unique_ratio": round(unique_ratio, 4),
            "corr_with_target": None if pd.isna(corr) else round(float(corr), 4),
            "maybe_id": unique_ratio >= threshold and col != target,
            "maybe_leakage": (corr is not None and not pd.isna(corr)
                              and abs(corr) >= threshold and col != target),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Агрегатор — полный отчёт по всем проверкам
# ---------------------------------------------------------------------------

def run_all_checks(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    group: str = GROUP_COL,
    numeric_cols: list[str] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Прогоняет все проверки и возвращает словарь с отчётами.
    Если verbose=True — печатает краткую сводку (удобно для ноутбука).
    """
    numeric_cols = numeric_cols or NUMERIC_COLS

    report = {
        "shape": check_shape(df),
        "dtypes": check_dtypes(df),
        "duplicates": check_duplicates(df),
        "id_candidates": check_id_columns(df),
        "missing_infinite": check_missing_and_infinite(df, numeric_cols),
        "impossible_values": check_impossible_values(df, numeric_cols),
        "geometric_consistency": check_geometric_consistency(df),
        "rare_categories": check_rare_categories(df, group) if group in df else None,
        "class_balance": check_class_balance(df, group) if group in df else None,
        "near_duplicates": check_near_duplicates(df, numeric_cols),
        "unique_leakage": check_unique_leakage_columns(df, target=target),
    }

    if verbose:
        print("=== SHAPE ===")
        print(report["shape"])
        print("\n=== DTYPES ===")
        print(report["dtypes"].to_string(index=False))
        print("\n=== DUPLICATES ===")
        print({k: v for k, v in report["duplicates"].items()
               if k != "duplicate_examples"})
        print("\n=== MISSING / INFINITE ===")
        print(report["missing_infinite"].to_string(index=False))
        print("\n=== IMPOSSIBLE VALUES ===")
        print(report["impossible_values"].to_string(index=False))
        print("\n=== GEOMETRIC CONSISTENCY ===")
        gc = report["geometric_consistency"]
        print("ok:", gc["ok"], "| violations:", gc.get("violations_count"))
        print("\n=== CLASS BALANCE ===")
        print(report["class_balance"])
        print("\n=== UNIQUE / LEAKAGE ===")
        print(report["unique_leakage"].to_string(index=False))

    return report
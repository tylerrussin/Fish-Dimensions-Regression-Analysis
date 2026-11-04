"""Модуль проверок качества данных для EDA платформы FishGrow."""

import pandas as pd
import numpy as np


def check_shape_and_types(df):
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def check_duplicates(df):
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_index": bool(df.index.is_unique),
        "index_duplicates": int(df.index.duplicated().sum()),
    }


def check_missing_and_inf(df):
    num = df.select_dtypes(include="number")
    return pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().sum() / len(df) * 100).round(2),
        "n_inf": num.apply(lambda s: np.isinf(s).sum()).reindex(df.columns, fill_value=0),
        "n_nonpositive": num.apply(lambda s: (s <= 0).sum()).reindex(df.columns, fill_value=0),
        "min": num.min().reindex(df.columns),
        "max": num.max().reindex(df.columns),
    })


def check_geometric_consistency(df):
    return {
        "Length1_lt_Length2": bool((df["Length1"] < df["Length2"]).all()),
        "Length2_lt_Length3": bool((df["Length2"] < df["Length3"]).all()),
        "all_positive": bool((df[["Length1", "Length2", "Length3", "Height", "Width"]] > 0).all().all()),
        "n_violations": int(
            (~(df["Length1"] < df["Length2"])).sum()
            + (~(df["Length2"] < df["Length3"])).sum()
        ),
    }


def check_rare_categories(df, col, min_count=5):
    vc = df[col].value_counts()
    return {
        "n_categories": int(vc.shape[0]),
        "counts": vc.to_dict(),
        "rare": vc[vc < min_count].to_dict(),
        "imbalance_ratio": round(vc.max() / vc.min(), 2) if len(vc) > 1 else None,
    }


def check_near_duplicates(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    sub = df[cols].round(6)
    return {"n_near_duplicates": int(sub.duplicated().sum())}


def check_unique_fields(df):
    result = {}
    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)
        result[col] = {
            "n_unique": int(n_unique),
            "unique_ratio": round(n_unique / len(df), 3),
            "is_unique": n_unique == len(df),
        }
    return result


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Формальная схема проверки данных FishGrow.

    Проверяет:
    - наличие ожидаемых столбцов
    - диапазоны значений
    - допустимые виды рыб
    - пропуски

    Возвращает:
    - valid: bool
    - errors: list[str]
    - n_rows, n_cols: размеры
    """
    expected_columns = ['Species', 'Weight', 'Length1', 'Length2',
                        'Length3', 'Height', 'Width']

    expected_ranges = {
        'Weight':  (1, 2000, 'граммы'),
        'Length1': (5, 100, 'см'),
        'Length2': (5, 100, 'см'),
        'Length3': (5, 100, 'см'),
        'Height':  (1, 30, 'см'),
        'Width':   (1, 15, 'см'),
    }

    expected_species = {'Bream', 'Roach', 'Perch', 'Pike',
                        'Smelt', 'Parkki', 'Whitefish'}

    errors = []

    missing = set(expected_columns) - set(df.columns)
    if missing:
        errors.append(f"Отсутствуют столбцы: {sorted(missing)}")

    for col, (lo, hi, units) in expected_ranges.items():
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < lo:
                errors.append(f"{col}: min={col_min} < {lo} ({units}) — возможна ошибка единиц")
            if col_max > hi:
                errors.append(f"{col}: max={col_max} > {hi} ({units}) — возможна ошибка единиц")

    if 'Species' in df.columns:
        unknown = set(df['Species'].unique()) - expected_species
        if unknown:
            errors.append(f"Неизвестные виды: {sorted(unknown)}")

        # 4. Проверка пропусков — только исходные поля
    original_columns = ['Species', 'Weight', 'Length1', 'Length2',
                        'Length3', 'Height', 'Width']
    existing_original = [c for c in original_columns if c in df.columns]
    missing_in_original = df[existing_original].isna().sum()
    if missing_in_original.sum() > 0:
        errors.append(f"Пропуски в исходных полях: {missing_in_original[missing_in_original > 0].to_dict()}")
        
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }
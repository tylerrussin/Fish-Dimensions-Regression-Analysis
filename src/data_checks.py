"""
Повторно используемые проверки качества данных для FishGrow.

Модуль используется в notebooks/01-eda.ipynb и последующих этапах проекта.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

# Основные списки колонок и целевая переменная
NUMERIC_COLS = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
TARGET = 'Weight'
GROUP = 'Species'
SCHEMA_PATH = Path(__file__).with_name('fish_schema.json')
EXPECTED_UNITS = {
    'Species': None,
    'Weight': 'g',
    'Length1': 'cm',
    'Length2': 'cm',
    'Length3': 'cm',
    'Height': 'cm',
    'Width': 'cm',
}


def load_data(path: str) -> pd.DataFrame:
    """Загрузка датасета с учётом UTF-8 BOM маркера."""
    return pd.read_csv(path, encoding='utf-8-sig')


def check_shape(df: pd.DataFrame) -> tuple:
    """Возвращает размерность таблицы (строки, столбцы)."""
    return df.shape


def check_dtypes(df: pd.DataFrame) -> pd.Series:
    """Возвращает типы данных по колонкам."""
    return df.dtypes.astype(str)


def check_duplicates(df: pd.DataFrame) -> int:
    """Подсчёт полного количества дубликатов строк."""
    return int(df.duplicated().sum())


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Подсчёт количества пропущенных значений по колонкам."""
    return df.isna().sum()


def check_infinity(df: pd.DataFrame, cols: list = None) -> dict:
    """Проверка наличия бесконечных значений (inf / -inf)."""
    cols = cols or NUMERIC_COLS
    return {c: int(np.isinf(df[c]).sum()) for c in cols}


def check_impossible(df: pd.DataFrame, cols: list = None) -> dict:
    """Проверка физически невозможных значений (<= 0)."""
    cols = cols or NUMERIC_COLS
    return {c: int((df[c] <= 0).sum()) for c in cols}


def check_geometric_consistency(df: pd.DataFrame) -> dict:
    """
    Проверка геометрической логики:
    - Length1 <= Length2 <= Length3
    - Height < Length3 и Width < Length3
    - Отсутствие нетипичных или неположительных значений.
    """
    return {
        'Length1 > Length2': int((df['Length1'] > df['Length2']).sum()),
        'Length2 > Length3': int((df['Length2'] > df['Length3']).sum()),
        'Height >= Length3': int((df['Height'] >= df['Length3']).sum()),
        'Width >= Length3': int((df['Width'] >= df['Length3']).sum()),
        'any non-positive': int((df[NUMERIC_COLS] <= 0).any(axis=1).sum()),
    }


def check_categories(df: pd.DataFrame, col: str = GROUP) -> pd.DataFrame:
    """Анализ категорий: численность, доли и выявление редких классов."""
    counts = df[col].value_counts()
    return pd.DataFrame({
        'count': counts,
        'share': (counts / len(df)).round(4),
        'is_rare': counts < 10,
    })


def find_near_duplicates(df: pd.DataFrame, features: list = None, decimals: int = 2, exclude_cols: list = None):
    """Поиск ближних дубликатов по совпадающим геометриям с округлением."""
    features = features or ['Length1', 'Length2', 'Length3', 'Height', 'Width']
    exclude_cols = exclude_cols or [TARGET]
    cols = [c for c in features if c not in exclude_cols]
    mask = df[cols].round(decimals).duplicated(keep=False)
    return df[mask], mask


def find_leakage_candidates(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Анализ колонок на риски утечки данных или свойства уникальных ID."""
    rows = []
    for col in df.columns:
        if col == target:
            continue
        n_unique = df[col].nunique(dropna=False)
        corr = np.nan
        if pd.api.types.is_numeric_dtype(df[col]):
            corr = df[[col, target]].corr().iloc[0, 1]
        rows.append({
            'column': col,
            'n_unique': int(n_unique),
            'unique_ratio': round(n_unique / len(df), 4),
            'corr_with_target': round(corr, 4) if not np.isnan(corr) else None,
            'id_like': n_unique == len(df),
            'high_corr': (not np.isnan(corr)) and abs(corr) > 0.95,
        })
    return pd.DataFrame(rows)


def full_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица первичного качества: типы, пропуски, уникальные и физические ошибки."""
    rows = []
    n = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        n_unique = int(df[col].nunique(dropna=False))
        n_impossible = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            n_impossible = int((df[col] <= 0).sum())
        rows.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'missing': n_missing,
            'missing_share': round(n_missing / n, 4),
            'unique': n_unique,
            'impossible': n_impossible,
        })
    return pd.DataFrame(rows)


def iqr_bounds(series: pd.Series, k: float = 1.5) -> dict:
    """Вычисление границ выбросов по межквартильному размаху (IQR)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower': Q1 - k * IQR,
        'upper': Q3 + k * IQR,
    }


def mad_zscore(series: pd.Series):
    """Вычисление устойчивых z-оценок через медиану и MAD."""
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index), med, mad
    z = 0.6745 * (series - med) / mad
    return z, med, mad


def outliers_iqr(series: pd.Series, k: float = 1.5):
    """Возвращает маску выбросов и границы по методу IQR."""
    b = iqr_bounds(series, k)
    mask = (series < b['lower']) | (series > b['upper'])
    return mask, b


def outliers_mad(series: pd.Series, threshold: float = 3.5):
    """Возвращает маску выбросов, z-оценки и параметры по методу MAD."""
    z, med, mad = mad_zscore(series)
    mask = z.abs() > threshold
    return mask, z, med, mad


def validate_fish_schema(df: pd.DataFrame, schema_path: Path = SCHEMA_PATH) -> list:
    """Проверяет таблицу против JSON-контракта и ожидаемых единиц измерения."""
    schema = json.loads(Path(schema_path).read_text(encoding='utf-8'))
    properties = schema['properties']
    errors = []

    missing_columns = sorted(set(schema['required']) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(properties))
    if missing_columns:
        errors.append(f'Missing columns: {missing_columns}')
    if extra_columns and schema.get('additionalProperties') is False:
        errors.append(f'Unexpected columns: {extra_columns}')

    for column, expected_unit in EXPECTED_UNITS.items():
        if column in properties and properties[column].get('x-unit') != expected_unit:
            errors.append(f'{column} unit must be {expected_unit!r}')

    for column in set(df.columns) & set(properties):
        field = properties[column]
        values = df[column]
        if values.isna().any() and not field.get('nullable', False):
            errors.append(f'{column} contains null values')

        if field['type'] == 'number':
            numeric = pd.to_numeric(values, errors='coerce')
            if numeric.isna().sum() > values.isna().sum():
                errors.append(f'{column} contains non-numeric values')
                continue
            finite = numeric.dropna()
            if not np.isfinite(finite).all():
                errors.append(f'{column} contains non-finite values')
                continue
            if 'minimum' in field and (finite < field['minimum']).any():
                errors.append(f'{column} is below minimum {field["minimum"]}')
            if 'maximum' in field and (finite > field['maximum']).any():
                errors.append(f'{column} is above maximum {field["maximum"]}')
            positive = finite[finite > 0]
            if 'x-min-positive' in field and not positive.empty and (positive < field['x-min-positive']).any():
                errors.append(f'{column} positive values are below x-min-positive {field["x-min-positive"]}')
        elif field['type'] == 'string':
            if not pd.api.types.is_string_dtype(values.dtype):
                errors.append(f'{column} must contain strings')
            allowed = field.get('enum')
            if allowed is not None and not values.dropna().isin(allowed).all():
                errors.append(f'{column} contains values outside the declared enum')

    return errors
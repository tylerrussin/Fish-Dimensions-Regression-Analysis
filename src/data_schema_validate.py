"""Проверка датасета на соответствие формальной схеме."""
from __future__ import annotations

import pandas as pd

from data_schema import FieldSchema, get_schema


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Проверяет df на соответствие схеме.

    Возвращает dict с результатами: {field_name: [list_of_errors]}.
    Пустой список = всё ок для поля.
    """
    results: dict[str, list[str]] = {}

    for schema in get_schema():
        errors: list[str] = []
        name = schema.name

        # 1. Поле присутствует?
        if name not in df.columns:
            errors.append(f"Поле отсутствует в датафрейме")
            results[name] = errors
            continue

        # 2. Пропуски
        n_missing = int(df[name].isna().sum())
        if n_missing > 0 and not schema.nullable:
            errors.append(f"Пропуски запрещены, найдено {n_missing}")

        # 3. Тип и диапазон для числовых
        if schema.dtype == "numeric":
            if not pd.api.types.is_numeric_dtype(df[name]):
                errors.append(f"Ожидался числовой тип, получен {df[name].dtype}")
            else:
                if schema.min_value is not None:
                    n_below = int((df[name] < schema.min_value).sum())
                    if n_below > 0:
                        errors.append(
                            f"{n_below} значений ниже минимума {schema.min_value} {schema.units or ''}"
                        )
                if schema.max_value is not None:
                    n_above = int((df[name] > schema.max_value).sum())
                    if n_above > 0:
                        errors.append(
                            f"{n_above} значений выше максимума {schema.max_value} {schema.units or ''}"
                        )
                # Эвристика на смену единиц: если max сильно ниже ожидаемого —
                # возможна смена единиц (кг вместо г, м вместо см)
                observed_max = float(df[name].max())
                if schema.max_value is not None and observed_max < schema.max_value * 0.05:
                    errors.append(
                        f"Максимум {observed_max:.2f} аномально низкий "
                        f"(<5% от ожидаемого {schema.max_value}) — возможна смена единиц"
                    )

        # 4. Категориальные значения
        if schema.dtype == "categorical" and schema.allowed_values:
            unknown = set(df[name].dropna().unique()) - set(schema.allowed_values)
            if unknown:
                errors.append(f"Неизвестные категории: {sorted(unknown)}")

        results[name] = errors

    return results


def is_valid(df: pd.DataFrame) -> bool:
    """True, если датасет полностью соответствует схеме."""
    results = validate_dataframe(df)
    return all(len(errs) == 0 for errs in results.values())

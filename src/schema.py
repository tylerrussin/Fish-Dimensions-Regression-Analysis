"""Формальная схема данных FishGrow.

Описывает типы, диапазоны и единицы измерения каждого признака.
Используется для автоматической проверки при загрузке данных
и в тестах (см. tests/test_schema.py).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class FieldSpec:
    """Описание одного поля данных."""
    name: str
    dtype: str
    unit: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list] = None
    nullable: bool = False


# Основная схема датасета Fish Market
SCHEMA = [
    FieldSpec(
        name="Species",
        dtype="category",
        unit="-",
        allowed_values=[
            "Bream", "Roach", "Whitefish", "Parkki",
            "Perch", "Pike", "Smelt",
        ],
    ),
    FieldSpec(name="Weight", dtype="float", unit="g",
              min_value=0.0, max_value=3000.0),
    FieldSpec(name="Length1", dtype="float", unit="cm",
              min_value=5.0, max_value=100.0),
    FieldSpec(name="Length2", dtype="float", unit="cm",
              min_value=5.0, max_value=100.0),
    FieldSpec(name="Length3", dtype="float", unit="cm",
              min_value=5.0, max_value=100.0),
    FieldSpec(name="Height", dtype="float", unit="cm",
              min_value=1.0, max_value=30.0),
    FieldSpec(name="Width", dtype="float", unit="cm",
              min_value=0.5, max_value=15.0),
]


def validate_schema(df, schema=SCHEMA):
    """Проверяет df на соответствие схеме.

    Возвращает список строк-ошибок. Пустой список = всё ок.
    """
    errors = []
    for spec in schema:
        if spec.name not in df.columns:
            errors.append(f"Отсутствует поле: {spec.name}")
            continue

        col = df[spec.name]

        if spec.min_value is not None:
            n_below = int((col < spec.min_value).sum())
            if n_below > 0:
                errors.append(
                    f"{spec.name}: {n_below} значений < "
                    f"{spec.min_value} {spec.unit}"
                )

        if spec.max_value is not None:
            n_above = int((col > spec.max_value).sum())
            if n_above > 0:
                errors.append(
                    f"{spec.name}: {n_above} значений > "
                    f"{spec.max_value} {spec.unit}"
                )

        if spec.allowed_values is not None:
            unexpected = set(col.dropna().unique()) - set(spec.allowed_values)
            if unexpected:
                errors.append(
                    f"{spec.name}: неожиданные значения {unexpected}"
                )

    return errors


def check_units(df) -> list:
    """Эвристики для проверки единиц измерения ключевых полей.

    Возвращает список предупреждений, если единицы похожи на изменённые.
    """
    warnings = []

    # Weight должен быть в граммах (максимум ~1650 г в исходных данных)
    if df["Weight"].max() < 100:
        warnings.append(
            f"Weight.max() = {df['Weight'].max():.2f}. "
            "Похоже, единицы изменились: г → кг?"
        )

    # Length3 должен быть в сантиметрах (максимум ~68 см)
    if df["Length3"].max() < 10:
        warnings.append(
            f"Length3.max() = {df['Length3'].max():.2f}. "
            "Похоже, единицы изменились: см → м?"
        )

    return warnings

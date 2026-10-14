"""Тесты для проверки схемы данных FishGrow.

Запуск: python -m pytest tests/ -v

Тесты падают, если:
- изменились единицы измерения (г → кг, см → м);
- значения вышли за допустимые диапазоны;
- появились новые виды или пропали старые;
- изменилось число колонок.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Добавляем src в путь, чтобы импортировать schema
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from schema import SCHEMA, validate_schema, check_units  # noqa: E402


@pytest.fixture
def df():
    """Загружает датасет для тестов."""
    return pd.read_csv(ROOT / "data" / "Fish.csv")


def test_columns_present(df):
    """Все колонки из схемы должны присутствовать в датасете."""
    expected = {spec.name for spec in SCHEMA}
    actual = set(df.columns)
    missing = expected - actual
    assert not missing, f"Отсутствуют колонки: {missing}"


def test_species_unchanged(df):
    """Набор видов должен совпадать с известным."""
    expected = {"Bream", "Roach", "Whitefish", "Parkki",
                "Perch", "Pike", "Smelt"}
    actual = set(df["Species"].unique())
    assert actual == expected, (
        f"Появились новые виды: {actual - expected}; "
        f"пропали: {expected - actual}"
    )


def test_weight_in_grams(df):
    """Масса должна быть в граммах.

    Если кто-то переведёт в килограммы, максимум упадёт
    с ~1650 до ~1.65 — тест сработает.
    """
    assert df["Weight"].max() > 100, (
        f"Weight.max() = {df['Weight'].max()}. "
        "Похоже, единицы изменились: г → кг?"
    )


def test_lengths_in_centimeters(df):
    """Длины должны быть в сантиметрах."""
    for col in ["Length1", "Length2", "Length3"]:
        assert df[col].max() > 10, (
            f"{col}.max() = {df[col].max()}. "
            "Похоже, единицы изменились: см → м?"
        )


def test_dimensions_reasonable(df):
    """Длины и высота должны быть в разумных диапазонах."""
    assert df["Length1"].between(5, 100).all(), "Length1 вне диапазона [5, 100]"
    assert df["Length2"].between(5, 100).all(), "Length2 вне диапазона [5, 100]"
    assert df["Length3"].between(5, 100).all(), "Length3 вне диапазона [5, 100]"
    assert df["Height"].between(1, 30).all(), "Height вне диапазона [1, 30]"
    assert df["Width"].between(0.5, 15).all(), "Width вне диапазона [0.5, 15]"


def test_no_negative_values(df):
    """Все числовые признаки должны быть неотрицательны."""
    numeric_cols = ["Weight", "Length1", "Length2", "Length3",
                    "Height", "Width"]
    for col in numeric_cols:
        assert (df[col] >= 0).all(), f"В {col} есть отрицательные значения"


def test_units_warnings_empty(df):
    """Эвристика по единицам не должна выдавать предупреждений."""
    warnings = check_units(df)
    assert not warnings, f"Предупреждения по единицам: {warnings}"


def test_schema_has_no_critical_errors(df):
    """Схема не должна содержать критических ошибок."""
    errors = validate_schema(df)
    assert not errors, f"Критические ошибки схемы: {errors}"

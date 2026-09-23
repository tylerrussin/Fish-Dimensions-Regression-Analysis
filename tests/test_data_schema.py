"""Быстрые проверки схемы и физического смысла датасета Fish.csv."""
from pathlib import Path

import pandas as pd
import pytest

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "data" / "Fish.csv"
)

EXPECTED_COLUMNS = [
    "Species", "Weight",
    "Length1", "Length2", "Length3",
    "Height", "Width",
]

NUMERIC_COLUMNS = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def test_file_exists():
    assert DATA_PATH.exists(), f"Датасет не найден: {DATA_PATH}"


def test_schema(df: pd.DataFrame):
    """Схема столбцов совпадает с ожидаемой."""
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Ожидалось: {EXPECTED_COLUMNS}\nПолучено:  {list(df.columns)}"
    )


def test_no_missing(df: pd.DataFrame):
    """В ключевых столбцах нет пропусков."""
    assert df[EXPECTED_COLUMNS].isna().sum().sum() == 0, "Найдены пропуски"


def test_positive_measurements(df: pd.DataFrame):
    """Все числовые измерения строго положительные."""
    for col in NUMERIC_COLUMNS:
        assert (df[col] > 0).all(), f"Неположительные значения в '{col}'"


def test_species_categories(df: pd.DataFrame):
    """Species без пустых строк и лишних пробелов."""
    s = df["Species"]
    assert s.notna().all(), "Species содержит пропуски"
    assert (s.str.strip() == s).all(), "Лишние пробелы в Species"
    assert (s.str.len() > 0).all(), "Пустые строки в Species"


def test_length_order_consistency(df: pd.DataFrame):
    """Length1 <= Length2 <= Length3 — обычный порядок трёх измерений."""
    assert (df["Length1"] <= df["Length2"] + 1e-6).all(), "Length1 > Length2"
    assert (df["Length2"] <= df["Length3"] + 1e-6).all(), "Length2 > Length3"

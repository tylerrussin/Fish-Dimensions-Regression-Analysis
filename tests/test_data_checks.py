"""Тесты для модуля src/data_checks.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest
from data_checks import validate_schema


@pytest.fixture
def valid_df():
    """Корректный DataFrame."""
    return pd.DataFrame({
        'Species': ['Bream', 'Roach', 'Pike'],
        'Weight':  [500.0, 150.0, 800.0],
        'Length1': [25.0, 18.0, 40.0],
        'Length2': [27.0, 20.0, 44.0],
        'Length3': [30.0, 22.0, 50.0],
        'Height':  [11.0, 6.0, 8.0],
        'Width':   [4.0, 3.0, 5.0],
    })


def test_valid_data_passes(valid_df):
    """Корректные данные должны проходить схему."""
    result = validate_schema(valid_df)
    assert result['valid'], f"Ошибка: {result['errors']}"


def test_weight_in_kg_fails():
    """Weight в килограммах (0.5 вместо 500) — схема должна сработать."""
    df = pd.DataFrame({
        'Species': ['Bream'], 'Weight': [0.5],
        'Length1': [25.0], 'Length2': [27.0], 'Length3': [30.0],
        'Height': [11.0], 'Width': [4.0],
    })
    result = validate_schema(df)
    assert not result['valid'], "Схема должна обнаружить Weight < 1 г"
    assert any('Weight' in err for err in result['errors'])


def test_length_in_meters_fails():
    """Length в метрах (0.3 вместо 30) — схема должна сработать."""
    df = pd.DataFrame({
        'Species': ['Bream'], 'Weight': [500.0],
        'Length1': [0.25], 'Length2': [0.27], 'Length3': [0.30],
        'Height': [11.0], 'Width': [4.0],
    })
    result = validate_schema(df)
    assert not result['valid'], "Схема должна обнаружить Length < 5 см"


def test_weight_out_of_range_fails():
    """Weight > 2000 г — ошибка диапазона."""
    df = pd.DataFrame({
        'Species': ['Pike'], 'Weight': [5000.0],
        'Length1': [25.0], 'Length2': [27.0], 'Length3': [30.0],
        'Height': [11.0], 'Width': [4.0],
    })
    result = validate_schema(df)
    assert not result['valid'], "Схема должна обнаружить Weight > 2000 г"


def test_unknown_species_fails():
    """Новый вид — ошибка схемы."""
    df = pd.DataFrame({
        'Species': ['Shark'], 'Weight': [500.0],
        'Length1': [25.0], 'Length2': [27.0], 'Length3': [30.0],
        'Height': [11.0], 'Width': [4.0],
    })
    result = validate_schema(df)
    assert not result['valid'], "Схема должна обнаружить неизвестный вид"


def test_missing_column_fails():
    """Пропущен столбец Height — ошибка схемы."""
    df = pd.DataFrame({
        'Species': ['Bream'], 'Weight': [500.0],
        'Length1': [25.0], 'Length2': [27.0], 'Length3': [30.0],
        'Width': [4.0],
    })
    result = validate_schema(df)
    assert not result['valid'], "Схема должна обнаружить отсутствующий столбец"


def test_missing_values_fail():
    """Пропуск в Weight — ошибка схемы."""
    df = pd.DataFrame({
        'Species': ['Bream'], 'Weight': [None],
        'Length1': [25.0], 'Length2': [27.0], 'Length3': [30.0],
        'Height': [11.0], 'Width': [4.0],
    })
    result = validate_schema(df)
    assert not result['valid'], "Схема должна обнаружить пропуски"
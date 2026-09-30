"""
Тесты соответствия датасета FishGrow формальной схеме.

Запуск:
    pytest tests/ -v

Срабатывают при:
- смене единиц ключевого поля (например, вес в кг вместо г);
- сужении / расширении диапазона значений;
- пропаже столбца;
- появлении неизвестного вида рыбы;
- отрицательных значениях.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from data_schema_validate import validate_dataframe, is_valid  # noqa: E402
from data_schema import get_schema_dict  # noqa: E402


DATA_PATH = ROOT / "assets" / "data" / "Fish.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def test_dataset_matches_schema(df):
    """Главный тест: датасет соответствует схеме."""
    result = validate_dataframe(df)
    errors = {k: v for k, v in result.items() if v}
    assert not errors, f"Датасет не соответствует схеме:\n{errors}"


def test_required_columns_present(df):
    """Все столбцы схемы присутствуют."""
    expected = set(get_schema_dict().keys())
    missing = expected - set(df.columns)
    assert not missing, f"Отсутствуют столбцы: {missing}"


def test_weight_units_grams(df):
    """
    Weight измеряется в граммах.
    Эвристика: если максимум < 10 — вероятно, вес в килограммах.
    Если максимум < 1 — вообще не масса рыбы.
    """
    w = df["Weight"].dropna()
    assert w.max() >= 100, (
        f"Максимум Weight = {w.max():.2f}. Ожидается > 100 г. "
        f"Возможно, единицы сменились на килограммы или что-то другое."
    )
    assert w.max() <= 3000, (
        f"Максимум Weight = {w.max():.2f} > 3000 г. "
        f"Слишком много для рыбы. Возможно, смена единиц."
    )


def test_length_units_centimeters(df):
    """Длины измеряются в сантиметрах. Ожидаемый максимум 20–200 см."""
    for col in ["Length1", "Length2", "Length3"]:
        m = df[col].dropna().max()
        assert 10 <= m <= 200, (
            f"{col}: максимум {m:.2f} см вне ожидаемого диапазона [10; 200]. "
            f"Возможно, единицы сменились на метры или миллиметры."
        )


def test_height_units_centimeters(df):
    """Height измеряется в сантиметрах."""
    m = df["Height"].dropna().max()
    assert 5 <= m <= 100, (
        f"Height: максимум {m:.2f} см вне диапазона [5; 100]. "
        f"Возможна смена единиц."
    )


def test_no_negative_values(df):
    """Физические величины неотрицательны."""
    numeric = df.select_dtypes("number")
    neg = (numeric < 0).any()
    assert not neg.any(), f"Отрицательные значения в: {neg[neg].index.tolist()}"


def test_species_known(df):
    """Все виды рыбы — из известного списка."""
    allowed = {"Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"}
    unknown = set(df["Species"].dropna().unique()) - allowed
    assert not unknown, (
        f"Неизвестные виды: {unknown}. "
        f"Если новый вид появился намеренно — обновите FISH_SCHEMA."
    )


def test_row_count_stable(df):
    """
    Количество строк соответствует ожидаемому.
    Если датасет обновился — тест напомнит пересчитать метрики.
    """
    assert len(df) == 159, (
        f"Ожидалось 159 строк, найдено {len(df)}. "
        f"Датасет изменился — проверьте §3.2 отчёта."
    )

import sys
from pathlib import Path

# Добавляем корень проекта в путь поиска модулей
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error

from src.train import load_and_split, build_preprocessor, add_volume


@pytest.fixture
def data_splits():
    """Фикстура для загрузки данных и деления на train/test."""
    return load_and_split()


def test_no_index_overlap(data_splits):
    """1. Индексы обучающей и тестовой выборок строго не должны пересекаться."""
    train_df, test_df = data_splits
    overlap = set(train_df.index).intersection(set(test_df.index))
    assert len(overlap) == 0, f"Обнаружено пересечение индексов между train и test: {overlap}"


def test_pipeline_handles_unknown_category():
    """2. Цепочка обработки корректно принимает строку с неизвестной категорией вида рыбы."""
    train_df, _ = load_and_split()
    features = ["Length1", "Length2", "Length3", "Height", "Width", "Species"]

    pipe = Pipeline([
        ("feat_eng", FunctionTransformer(add_volume)),
        ("prep", build_preprocessor(["Length3", "Height", "Width", "V_proxy"], ["Species"])),
        ("reg", Ridge(alpha=10.0))
    ])
    pipe.fit(train_df[features], train_df["Weight"])

    # Подаём особь с вымышленным видом, которого не было при обучении
    new_unknown_fish = pd.DataFrame([{
        "Length1": 30.0,
        "Length2": 32.0,
        "Length3": 35.0,
        "Height": 12.0,
        "Width": 5.0,
        "Species": "AlienFish"
    }])

    pred = pipe.predict(new_unknown_fish)
    assert len(pred) == 1, "Модель должна вернуть ровно одно предсказание"
    assert np.isfinite(pred[0]), "Предсказание должно быть конечным валидным числом"


def test_predictions_count_and_finiteness(data_splits):
    """3. Число прогнозов равно числу строк на входе, все значения конечны (нет NaN и Inf)."""
    train_df, test_df = data_splits
    features = ["Length1", "Length2", "Length3", "Height", "Width", "Species"]

    pipe = Pipeline([
        ("feat_eng", FunctionTransformer(add_volume)),
        ("prep", build_preprocessor(["Length3", "Height", "Width", "V_proxy"], ["Species"])),
        ("reg", Ridge(alpha=10.0))
    ])
    pipe.fit(train_df[features], train_df["Weight"])
    preds = pipe.predict(test_df[features])

    assert len(preds) == len(test_df), "Длина массива предсказаний не совпадает с числом строк теста"
    assert np.all(np.isfinite(preds)), "Обнаружены бесконечности или пропуски (NaN) в предсказаниях"


def test_manual_metrics_calculation():
    """4. Метрика MAE на контрольном примере совпадает с расчётом вручную."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])

    # Ручной расчет MAE: (|100 - 110| + |200 - 190| + |300 - 330|) / 3 = (10 + 10 + 30) / 3 = 50 / 3 = 16.666...
    expected_mae = (10.0 + 10.0 + 30.0) / 3.0
    calculated_mae = mean_absolute_error(y_true, y_pred)

    assert np.isclose(expected_mae, calculated_mae, atol=1e-5), "Расчёт метрики расходится с формулой"
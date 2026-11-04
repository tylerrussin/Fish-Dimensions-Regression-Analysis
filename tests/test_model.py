"""Тесты для модели FishGrow (Лабораторная №2)."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

# Добавляем src в path, чтобы импортировать train.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train import build_preprocessor, load_config, load_data, split_data


@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def data():
    return load_data()


def test_indices_no_overlap(config, data):
    """1. Индексы train и test не пересекаются."""
    train, val, test = split_data(data, config)
    assert set(train.index).isdisjoint(set(test.index))
    assert set(train.index).isdisjoint(set(val.index))
    assert set(val.index).isdisjoint(set(test.index))


def test_pipeline_unknown_category(config):
    """2. Цепочка обработки принимает строку с неизвестной категорией."""
    X_train = pd.DataFrame({
        "Length1": [10.0, 20.0], "Length2": [11.0, 21.0], "Length3": [12.0, 22.0],
        "Height": [2.0, 3.0], "Width": [1.0, 2.0],
        "Species": ["Bream", "Roach"]
    })
    
    preprocess = build_preprocessor(
        config["features"]["numeric"], 
        config["features"]["categorical"]
    )
    preprocess.fit(X_train)
    
    # Новый вид — "Shark" (не было в train)
    X_test = pd.DataFrame({
        "Length1": [15.0], "Length2": [16.0], "Length3": [17.0],
        "Height": [2.5], "Width": [1.5],
        "Species": ["Shark"]
    })
    
    # Если не упадет с ошибкой — тест пройден
    X_transformed = preprocess.transform(X_test)
    assert X_transformed.shape[0] == 1


def test_predictions_finite_and_length(config, data):
    """3. Число прогнозов равно числу входных строк, все значения конечны."""
    train, val, test = split_data(data, config)
    numeric_cols = config["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]
    target = config["data"]["target"]
    
    X_train = train[numeric_cols + categorical_cols]
    y_train = train[target]
    X_test = test[numeric_cols + categorical_cols]
    
    # Используем DummyRegressor для скорости
    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    assert len(preds) == len(X_test)
    assert np.all(np.isfinite(preds))


def test_metrics_manual_calculation():
    """4. Метрики на зафиксированном примере совпадают с ручным расчётом."""
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    # Ошибки: |10-12| = 2, |20-18| = 2. Среднее MAE = 2.0
    mae = mean_absolute_error(y_true, y_pred)
    assert mae == 2.0


def test_model_save_load(tmp_path, config):
    """5. Сохранённая цепочка после загрузки воспроизводит прогноз."""
    X_train = pd.DataFrame({
        "Length1": [10.0], "Length2": [11.0], "Length3": [12.0],
        "Height": [2.0], "Width": [1.0], "Species": ["Bream"]
    })
    y_train = [100.0]
    
    preprocess = build_preprocessor(
        config["features"]["numeric"], 
        config["features"]["categorical"]
    )
    model = DummyRegressor()
    pipeline = Pipeline([("preprocess", preprocess), ("regressor", model)])
    pipeline.fit(X_train, y_train)
    
    # Сохраняем во временную папку (tmp_path)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    
    # Загружаем
    loaded = joblib.load(model_path)
    pred1 = pipeline.predict(X_train)
    pred2 = loaded.predict(X_train)
    
    assert np.allclose(pred1, pred2)
"""
train.py — обучение моделей E0–E4 для FishGrow.

Запуск:
    python src/train.py
"""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor


# === Пути ===
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "assets" / "data" / "Fish.csv"
CONFIG_PATH = ROOT / "src" / "config.yaml"
MODELS_DIR = ROOT / "models"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Загрузить конфиг из YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Загрузить данные."""
    df = pd.read_csv(path)
    return df


def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Разбить данные: train / val / test (60/20/20) со стратификацией по Species.
    """
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    random_state = config["data"]["random_state"]

    train, temp = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=df["Species"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=random_state,
        stratify=temp["Species"],
    )
    return train, val, test

def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Единая цепочка предобработки:
    - числовые: заполнение медианой + масштабирование
    - категориальные: заполнение модой + one-hot
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])

def evaluate_model(model, X_test, y_test) -> dict:
    """Посчитать MSE, MAE, R² на тестовой выборке."""
    prediction = model.predict(X_test)
    return {
        "mse": float(mean_squared_error(y_test, prediction)),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "r2": float(r2_score(y_test, prediction)),
        "prediction": prediction,
    }

def run_experiment(
    name: str,
    model,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    config: dict,
    use_mlflow: bool = True,
) -> dict:
    """
    Обучить модель, посчитать метрики, залогировать в MLflow.
    Возвращает словарь с метриками и моделью.
    """
    print(f"\n=== {name} ===")

    # Обучаем
    model.fit(X_train, y_train)

    # Метрики на val и test
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    print(f"Val  — MSE: {val_metrics['mse']:.2f}, MAE: {val_metrics['mae']:.2f}, R²: {val_metrics['r2']:.4f}")
    print(f"Test — MSE: {test_metrics['mse']:.2f}, MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.4f}")

    # MLflow
    if use_mlflow:
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model": model.__class__.__name__,
                "random_state": config["data"]["random_state"],
                "test_size": config["data"]["test_size"],
                "val_size": config["data"]["val_size"],
            })
            mlflow.log_metrics({
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
            })

            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="cloudpickle",
            )

    return {
        "name": name,
        "model": model,
        "val": val_metrics,
        "test": test_metrics,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true", help="Без MLflow")
    args = parser.parse_args()

    config = load_config()
    df = load_data()
    train, val, test = split_data(df, config)

    print(f"Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
    print(f"Всего: {len(df)}")

    target = config["data"]["target"]
    numeric_cols = config["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]

    X_train = train[numeric_cols + categorical_cols]
    y_train = train[target]
    X_val = val[numeric_cols + categorical_cols]
    y_val = val[target]
    X_test = test[numeric_cols + categorical_cols]
    y_test = test[target]

    use_mlflow = not args.no_mlflow

    results = []

    # === E0: наивная (среднее) ===

    results.append(run_experiment(
    "E0_naive_mean",
    DummyRegressor(strategy="mean"),
    X_train, y_train, X_val, y_val, X_test, y_test,
    config, use_mlflow,
    ))

    # === E1: линейная регрессия ===
    preprocess = build_preprocessor(numeric_cols, categorical_cols)
    lr = Pipeline([
        ("preprocess", preprocess),
        ("regressor", LinearRegression()),
    ])
    results.append(run_experiment(
        "E1_linear",
        lr,
        X_train, y_train, X_val, y_val, X_test, y_test,
        config, use_mlflow,
    ))

    # === E2: Ridge с подбором alpha по CV ===
    preprocess = build_preprocessor(numeric_cols, categorical_cols)
    ridge = Pipeline([
        ("preprocess", preprocess),
        ("regressor", Ridge()),
    ])
    param_grid = {"regressor__alpha": config["experiments"]["E2"]["alpha_grid"]}
    kfold = KFold(n_splits=5, shuffle=True, random_state=config["data"]["random_state"])
    grid = GridSearchCV(ridge, param_grid, cv=kfold, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)
    print(f"\nRidge: лучший alpha = {grid.best_params_['regressor__alpha']}")
    results.append(run_experiment(
        "E2_ridge_best",
        grid.best_estimator_,
        X_train, y_train, X_val, y_val, X_test, y_test,
        config, use_mlflow,
    ))

    # === E3: Random Forest ===
    preprocess = build_preprocessor(numeric_cols, categorical_cols)
    rf = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(random_state=config["data"]["random_state"])),
    ])
    param_grid = {
        "regressor__n_estimators": config["experiments"]["E3"]["n_estimators_grid"],
        "regressor__max_depth": config["experiments"]["E3"]["max_depth_grid"],
    }
    grid = GridSearchCV(rf, param_grid, cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"\nRF: лучшие параметры = {grid.best_params_}")
    results.append(run_experiment(
        "E3_random_forest",
        grid.best_estimator_,
        X_train, y_train, X_val, y_val, X_test, y_test,
        config, use_mlflow,
    ))

    # === E4: Ridge с V_approx ===
    train_E4 = train.copy()
    val_E4 = val.copy()
    test_E4 = test.copy()
    for d in [train_E4, val_E4, test_E4]:
        d["V_approx"] = d["Length3"] * d["Height"] * d["Width"]

    numeric_E4 = config["experiments"]["E4"]["features"]["numeric"]
    X_train_E4 = train_E4[numeric_E4 + categorical_cols]
    y_train_E4 = train_E4[target]
    X_val_E4 = val_E4[numeric_E4 + categorical_cols]
    y_val_E4 = val_E4[target]
    X_test_E4 = test_E4[numeric_E4 + categorical_cols]
    y_test_E4 = test_E4[target]

    preprocess = build_preprocessor(numeric_E4, categorical_cols)
    ridge_E4 = Pipeline([
        ("preprocess", preprocess),
        ("regressor", Ridge(alpha=grid.best_params_["regressor__alpha"] if "regressor__alpha" in grid.best_params_ else 1.0)),
    ])
    results.append(run_experiment(
        "E4_ridge_volume",
        ridge_E4,
        X_train_E4, y_train_E4,
        X_val_E4, y_val_E4,
        X_test_E4, y_test_E4,
        config, use_mlflow,
    ))

    # === Итоговая таблица ===
    print("\n=== ИТОГОВАЯ ТАБЛИЦА ===")
    print(f"{'Модель':<25} {'Val MAE':>10} {'Val R²':>10} {'Test MAE':>10} {'Test R²':>10}")
    for r in results:
        print(f"{r['name']:<25} {r['val']['mae']:>10.2f} {r['val']['r2']:>10.4f} {r['test']['mae']:>10.2f} {r['test']['r2']:>10.4f}")
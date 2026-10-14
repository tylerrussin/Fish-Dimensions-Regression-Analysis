from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATA_PATH = Path("data/Fish.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split():
  df = pd.read_csv(DATA_PATH)

  # Исключаем строку 40 (Roach с Weight = 0), так как это логическая ошибка
  df_clean = df[df["Weight"] > 0].copy().reset_index(drop=True)

  # Стратифицированное разбиение: 80% train, 20% test
  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  train_idx, test_idx = next(sss.split(df_clean, df_clean["Species"]))

  train_df = df_clean.iloc[train_idx].copy()
  test_df = df_clean.iloc[test_idx].copy()

  Path("data").mkdir(exist_ok=True)
  test_df.to_csv("data/test_frozen.csv", index=False)

  return train_df, test_df


def build_preprocessor(numeric_cols, categorical_cols):
  numeric_pipeline = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
  ])

  categorical_pipeline = Pipeline([
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("onehot", OneHotEncoder(handle_unknown="ignore")),
  ])

  return ColumnTransformer([
      ("num", numeric_pipeline, numeric_cols),
      ("cat", categorical_pipeline, categorical_cols),
  ])


def evaluate_cv(pipeline, X, y, cv_splits=5):
  """Кросс-валидация внутри train выборки (без подглядывания в test)"""
  kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
  mses, maes, r2s = [], [], []

  for tr_i, val_i in kf.split(X):
    X_tr, X_val = X.iloc[tr_i], X.iloc[val_i]
    y_tr, y_val = y.iloc[tr_i], y.iloc[val_i]

    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_val)

    mses.append(mean_squared_error(y_val, preds))
    maes.append(mean_absolute_error(y_val, preds))
    r2s.append(r2_score(y_val, preds))

  return {
      "cv_mse_mean": float(np.mean(mses)),
      "cv_mse_std": float(np.std(mses)),
      "cv_mae_mean": float(np.mean(maes)),
      "cv_mae_std": float(np.std(maes)),
      "cv_r2_mean": float(np.mean(r2s)),
      "cv_r2_std": float(np.std(r2s)),
  }


def plot_residuals(y_true, y_pred, exp_name):
  """Построение графиков: факт vs прогноз и анализ остатков"""
  fig, ax = plt.subplots(1, 2, figsize=(12, 5))
  residuals = y_true - y_pred

  # График 1: Факт vs Прогноз
  ax[0].scatter(y_true, y_pred, alpha=0.7, edgecolors="k")
  min_val = min(min(y_true), min(y_pred))
  max_val = max(max(y_true), max(y_pred))
  ax[0].plot(
      [min_val, max_val], [min_val, max_val], "r--", label="Идеальный прогноз"
  )
  ax[0].set_xlabel("Истинная масса (г)")
  ax[0].set_ylabel("Предсказанная масса (г)")
  ax[0].set_title(f"{exp_name}: Факт vs Прогноз")
  ax[0].grid(True)
  ax[0].legend()

  # График 2: Анализ остатков
  ax[1].scatter(y_pred, residuals, alpha=0.7, edgecolors="k", color="orange")
  ax[1].axhline(0, color="red", linestyle="--")
  ax[1].set_xlabel("Предсказанная масса (г)")
  ax[1].set_ylabel("Остаток (y - ŷ)")
  ax[1].set_title(f"{exp_name}: Анализ остатков")
  ax[1].grid(True)

  fig_path = FIGURES_DIR / f"{exp_name}_residuals.png"
  plt.tight_layout()
  plt.savefig(fig_path, dpi=120)
  plt.close()
  return fig_path


def run_experiment(
    exp_name, pipeline, X_train, y_train, X_test, y_test, params=None
):
  """Запуск эксперимента и сохранение всех артефактов в MLflow"""
  with mlflow.start_run(run_name=exp_name):
    if params:
      mlflow.log_params(params)
    mlflow.set_tag("dataset_version", "Fish.csv_v1")

    # 1. Считаем метрики на кросс-валидации
    cv_metrics = evaluate_cv(pipeline, X_train, y_train)
    mlflow.log_metrics(cv_metrics)

    # 2. Обучаем на всем train и оцениваем на test
    pipeline.fit(X_train, y_train)
    test_preds = pipeline.predict(X_test)

    test_metrics = {
        "test_mse": float(mean_squared_error(y_test, test_preds)),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
    }
    mlflow.log_metrics(test_metrics)

    # 3. Сохраняем графики и саму модель
    fig_file = plot_residuals(y_test, test_preds, exp_name)
    mlflow.log_artifact(str(fig_file))
    mlflow.sklearn.log_model(
    pipeline,
    name="model",
    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
)

    print(
        f"[{exp_name}] CV R2: {cv_metrics['cv_r2_mean']:.3f} | Test R2:"
        f" {test_metrics['test_r2']:.3f} | Test MAE:"
        f" {test_metrics['test_mae']:.1f} г"
    )
    return {**cv_metrics, **test_metrics}


# Функция добавления объема для эксперимента E4
def add_volume(df_in):
  df_out = df_in.copy()
  df_out["V_proxy"] = df_out["Length3"] * df_out["Height"] * df_out["Width"]
  return df_out


def main():
  mlflow.set_experiment("FishGrow_Regression")
  train_df, test_df = load_and_split()

  features = ["Length1", "Length2", "Length3", "Height", "Width", "Species"]
  X_tr, y_tr = train_df[features], train_df["Weight"]
  X_te, y_te = test_df[features], test_df["Weight"]

  num_cols = ["Length1", "Length2", "Length3", "Height", "Width"]
  cat_cols = ["Species"]

  print("=== ЗАПУСК ЭКСПЕРИМЕНТОВ MLFLOW ===")

  # E0. Базовый наивный прогноз (константное среднее)
  pipe_e0 = Pipeline([("dummy", DummyRegressor(strategy="mean"))])
  run_experiment("E0_dummy_mean", pipe_e0, X_tr, y_tr, X_te, y_te)

  # E1. Линейная регрессия (OLS)
  pipe_e1 = Pipeline([
      ("prep", build_preprocessor(num_cols, cat_cols)),
      ("reg", LinearRegression()),
  ])
  run_experiment("E1_linear_regression", pipe_e1, X_tr, y_tr, X_te, y_te)

  # E2. Гребневая регрессия (Ridge, alpha=10.0)
  pipe_e2 = Pipeline([
      ("prep", build_preprocessor(num_cols, cat_cols)),
      ("reg", Ridge(alpha=10.0)),
  ])
  run_experiment(
      "E2_ridge_alpha10",
      pipe_e2,
      X_tr,
      y_tr,
      X_te,
      y_te,
      params={"alpha": 10.0},
  )

  # E3. Нелинейная модель (Градиентный бустинг)
  pipe_e3 = Pipeline([
      ("prep", build_preprocessor(num_cols, cat_cols)),
      (
          "reg",
          HistGradientBoostingRegressor(
              max_iter=100, max_depth=4, random_state=42
          ),
      ),
  ])
  run_experiment(
      "E3_hist_gradient_boosting",
      pipe_e3,
      X_tr,
      y_tr,
      X_te,
      y_te,
      params={"max_iter": 100, "max_depth": 4},
  )

  # E4. Регрессия с физическим признаком объема V_proxy
  num_cols_e4 = ["Length3", "Height", "Width", "V_proxy"]
  pipe_e4 = Pipeline([
      ("feat_eng", FunctionTransformer(add_volume)),
      ("prep", build_preprocessor(num_cols_e4, cat_cols)),
      ("reg", Ridge(alpha=10.0)),
  ])
  run_experiment(
      "E4_ridge_with_volume",
      pipe_e4,
      X_tr,
      y_tr,
      X_te,
      y_te,
      params={"alpha": 10.0, "feature": "V_proxy"},
  )

  print("=== ВСЕ ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ ===")


if __name__ == "__main__":
  main()
"""
bootstrap.py — доверительные интервалы для MAE методом Bootstrap.
Запуск: python src/bootstrap.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from train import build_preprocessor, load_config, load_data, split_data

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "reports" / "figures" / "lab02"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    config = load_config()
    df = load_data()
    train, val, test = split_data(df, config)

    # Готовим данные для E4
    for d in [train, val, test]:
        d["V_approx"] = d["Length3"] * d["Height"] * d["Width"]

    target = config["data"]["target"]
    numeric_cols = config["experiments"]["E4"]["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]

    X_train = train[numeric_cols + categorical_cols]
    y_train = train[target]
    X_test = test[numeric_cols + categorical_cols]
    y_test = test[target]

    # Обучаем лучшую модель (E4)
    preprocess = build_preprocessor(numeric_cols, categorical_cols)
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", Ridge(alpha=0.1)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ====== BOOTSTRAP ======
    np.random.seed(42)
    n_iterations = 1000
    n_size = len(y_test)
    mae_scores = []

    print(f"Запуск Bootstrap ({n_iterations} итераций)...")
    for i in range(n_iterations):
        # Выборка с возвращением (индексы)
        indices = np.random.choice(n_size, n_size, replace=True)
        y_true_sample = y_test.iloc[indices]
        y_pred_sample = y_pred[indices]
        # Считаем MAE для этой подвыборки
        mae = mean_absolute_error(y_true_sample, y_pred_sample)
        mae_scores.append(mae)

    # Считаем 95% доверительный интервал
    lower_bound = np.percentile(mae_scores, 2.5)
    upper_bound = np.percentile(mae_scores, 97.5)
    mean_mae = np.mean(mae_scores)

    print(f"\nОбычный MAE на тесте: {mean_absolute_error(y_test, y_pred):.2f} г")
    print(f"Средний MAE по Bootstrap: {mean_mae:.2f} г")
    print(f"95% доверительный интервал: [{lower_bound:.2f} г, {upper_bound:.2f} г]")

    # ====== ИНТЕРВАЛ ПРОГНОЗА ======
    residuals = y_test - y_pred
    std_residuals = np.std(residuals)
    # Приблизительный 95% интервал прогноза для отдельного объекта
    pred_interval = 1.96 * std_residuals
    print(f"Ширина 95% интервала прогноза для одной рыбы: ± {pred_interval:.2f} г")

    # ====== ГРАФИК ======
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mae_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(lower_bound, color='red', linestyle='--', label=f'2.5% = {lower_bound:.2f}')
    ax.axvline(upper_bound, color='red', linestyle='--', label=f'97.5% = {upper_bound:.2f}')
    ax.axvline(mean_mae, color='green', linestyle='-', label=f'Mean = {mean_mae:.2f}')
    ax.set_title("Распределение MAE при Bootstrap (1000 итераций)")
    ax.set_xlabel("MAE, г")
    ax.set_ylabel("Частота")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bootstrap_mae.png", dpi=150, bbox_inches="tight")
    print(f"График сохранён: {FIG_DIR / 'bootstrap_mae.png'}")
    # plt.show()

if __name__ == "__main__":
    main()
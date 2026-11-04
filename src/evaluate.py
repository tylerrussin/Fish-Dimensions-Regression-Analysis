"""
evaluate.py — анализ ошибок лучшей модели (E4).

Запуск:
    python src/evaluate.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    target = config["data"]["target"]
    numeric_cols = config["experiments"]["E4"]["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]

    # V_approx
    for d in [train, val, test]:
        d["V_approx"] = d["Length3"] * d["Height"] * d["Width"]

    X_train = train[numeric_cols + categorical_cols]
    y_train = train[target]
    X_test = test[numeric_cols + categorical_cols]
    y_test = test[target]

    preprocess = build_preprocessor(numeric_cols, categorical_cols)
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", Ridge(alpha=0.1)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²:  {r2:.4f}")

    residuals = y_test - y_pred

    # График остатков
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolor="black")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Прогноз, г")
    axes[0].set_ylabel("Остаток (y_true − y_pred), г")
    axes[0].set_title("Остатки vs прогноз")

    axes[1].hist(residuals, bins=20, edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Остаток, г")
    axes[1].set_ylabel("Частота")
    axes[1].set_title("Распределение остатков")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"График сохранён: {FIG_DIR / 'residuals.png'}")

    # 5 крупнейших ошибок
    test_copy = test.copy()
    test_copy["y_true"] = y_test.values
    test_copy["y_pred"] = y_pred
    test_copy["abs_error"] = np.abs(test_copy["y_true"] - test_copy["y_pred"])

    top5 = test_copy.nlargest(5, "abs_error")
    print("\n=== 5 крупнейших ошибок ===")
    print(top5[["Species", "Weight", "y_pred", "abs_error"]].to_string())

    # MAE по видам
    print("\n=== MAE по видам ===")
    by_species = (
        test_copy.groupby("Species")
        .apply(lambda g: mean_absolute_error(g["y_true"], g["y_pred"]), include_groups=False)
        .sort_values(ascending=False)
    )
    print(by_species.to_string())

    # MAE по диапазонам размера (по Length3)
    print("\n=== MAE по диапазонам Length3 ===")
    test_copy["size_range"] = pd.cut(
        test_copy["Length3"],
        bins=[0, 25, 40, 100],
        labels=["мелкая (<25 см)", "средняя (25-40 см)", "крупная (>40 см)"]
    )
    mae_by_size = (
        test_copy.groupby("size_range", observed=True)
        .apply(lambda g: mean_absolute_error(g["y_true"], g["y_pred"]), include_groups=False)
    )
    print(mae_by_size.to_string())

    # Физически невозможные прогнозы
    neg = (y_pred < 0).sum()
    print(f"\nОтрицательных прогнозов: {neg}")


if __name__ == "__main__":
    main()
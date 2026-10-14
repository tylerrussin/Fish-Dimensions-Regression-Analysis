import sys
from pathlib import Path

# Добавляем корень проекта в пути поиска модулей Python
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer
from src.train import load_and_split, build_preprocessor, add_volume

def evaluate_best_model():
    # 1. Загружаем то самое замороженное разбиение
    train_df, test_df = load_and_split()
    
    features = ["Length1", "Length2", "Length3", "Height", "Width", "Species"]
    X_tr, y_tr = train_df[features], train_df["Weight"]
    X_te, y_te = test_df[features], test_df["Weight"]

    # 2. Обучаем нашу лучшую модель (E4 - Ridge с признаком объема)
    num_cols_e4 = ["Length3", "Height", "Width", "V_proxy"]
    cat_cols = ["Species"]
    
    best_pipeline = Pipeline([
        ("feat_eng", FunctionTransformer(add_volume)),
        ("prep", build_preprocessor(num_cols_e4, cat_cols)),
        ("reg", Ridge(alpha=10.0))
    ])
    
    best_pipeline.fit(X_tr, y_tr)
    preds = best_pipeline.predict(X_te)
    
    # 3. Добавляем предсказания в тестовую таблицу
    df_eval = test_df.copy()
    df_eval["Pred_Weight"] = np.round(preds, 1)
    df_eval["Abs_Error"] = np.round(np.abs(df_eval["Weight"] - df_eval["Pred_Weight"]), 1)
    df_eval["Residual"] = np.round(df_eval["Weight"] - df_eval["Pred_Weight"], 1)

    print("==================================================")
    print("        ОТЧЕТ ПО АНАЛИЗУ ОШИБОК МОДЕЛИ (E4)       ")
    print("==================================================")

    # Пункт 1: Проверка физически невозможных предсказаний
    negative_preds = df_eval[df_eval["Pred_Weight"] <= 0]
    print(f"\n1. Проверка отрицательных/нулевых предсказаний:")
    if len(negative_preds) == 0:
        print("   -> Физически невозможных (<= 0) предсказаний НЕТ. Модель ведет себя корректно.")
    else:
        print(f"   -> ВНИМАНИЕ: Найдено {len(negative_preds)} отрицательных прогнозов!")

    # Пункт 2: Топ-5 крупнейших ошибок
    print("\n2. Топ-5 крупнейших абсолютных ошибок:")
    cols_to_show = ["Species", "Weight", "Pred_Weight", "Abs_Error", "Length3", "Height", "Width"]
    top5 = df_eval.sort_values(by="Abs_Error", ascending=False).head(5)
    print(top5[cols_to_show].to_string(index=True))

    # Пункт 3: Средняя ошибка (MAE) по каждому виду рыб
    print("\n3. MAE по видам рыб:")
    species_mae = df_eval.groupby("Species")["Abs_Error"].mean().round(1).reset_index()
    species_mae.columns = ["Вид рыбы", "MAE (грамм)"]
    print(species_mae.to_string(index=False))

    # Пункт 4: Ошибка по размерным категориям
    print("\n4. MAE по весовым категориям:")
    # Делим рыбу на: мелкая (< 150г), средняя (150-500г), крупная (> 500г)
    bins = [0, 150, 500, 3000]
    labels = ["Мелкая (< 150г)", "Средняя (150-500г)", "Крупная (> 500г)"]
    df_eval["Weight_Category"] = pd.cut(df_eval["Weight"], bins=bins, labels=labels)
    cat_mae = df_eval.groupby("Weight_Category", observed=False)["Abs_Error"].mean().round(1).reset_index()
    cat_mae.columns = ["Категория рыбы", "MAE (грамм)"]
    print(cat_mae.to_string(index=False))

if __name__ == "__main__":
    evaluate_best_model()
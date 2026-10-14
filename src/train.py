from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATA_PATH = Path("data/Fish.csv")


def load_and_split():
    """Загрузка данных и стратифицированное разделение на Train (80%) и Test (20%)."""
    df = pd.read_csv(DATA_PATH)

    # Исключаем строку 40 (Roach с Weight = 0), так как это логическая ошибка
    df_clean = df[df["Weight"] > 0].copy().reset_index(drop=True)

    # Стратифицированное разбиение: сохраняет пропорции каждого вида рыбы
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(df_clean, df_clean["Species"]))

    train_df = df_clean.iloc[train_idx].copy()
    test_df = df_clean.iloc[test_idx].copy()

    # Сохраняем замороженный тестовый датасет, чтобы к нему больше не прикасаться до проверки
    Path("data").mkdir(exist_ok=True)
    test_df.to_csv("data/test_frozen.csv", index=False)

    print(
        f"Данные успешно разделены: Train = {len(train_df)} строк, Test = {len(test_df)} строк."
    )
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_and_split()
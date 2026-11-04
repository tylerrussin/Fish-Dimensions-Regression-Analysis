# Паспорт модели FishGrow

## Назначение
Прогноз массы рыбы по геометрическим измерениям и виду.

## Данные
- Источник: `assets/data/Fish.csv`
- Объём: 159 строк, 7 полей
- Целевая: `Weight` (граммы)

## Предобработка
- Числовые: SimpleImputer (медиана) + StandardScaler
- Категориальные: SimpleImputer (мода) + OneHotEncoder (handle_unknown='ignore')

## Лучшая модель: E4 (Ridge + V_approx)
- Класс: `sklearn.linear_model.Ridge(alpha=0.1)`
- Признаки: Length3, Height, Width, V_approx = Length3·Height·Width, Species
- Test MAE: 36.54 г
- Test R²: 0.9862

## Ограничения
- Отрицательные прогнозы (линейная модель).
- Whitefish (мало данных).
- Ошибка `Weight = 0` в исходных данных.

## Воспроизводимость
- `random_state=42`
- Разбиение: 60/20/20, стратификация
- MLflow run: `E4_ridge_volume`
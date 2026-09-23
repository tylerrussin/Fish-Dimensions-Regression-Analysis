## Источник
- URL: https://github.com/tylerrussin/Fish-Dimensions-Regression-Analysis.git
- Исходный коммит: 1f65547
- Дата: <22.09.2026>
- Ветка: lab/01-eda

## Структура
- run.py: есть
- app.py: есть
- pages/: есть
- assets/: есть
- Pipfile / Pipfile.lock: есть

## Хост-окружение
- ОС: macOS
- Python: 3.13.15 (python.org)
- Путь: /Library/Frameworks/Python.framework/Versions/3.13/bin/python3

## Установка зависимостей

### Исходная ошибка
Pipfile требовал python_version = "3.7".
Команда `pipenv sync --dev` завершилась ошибкой:
Warning: Python 3.7 was not found on your system...
Полный вывод сохранён в reports/lab01-step1-errors.txt.

### Минимальное изменение (вариант B)
- Файл: Pipfile
- Было: python_version = "3.7"
- Стало: python_version = "3.13"
- Дополнительно: pipenv lock (пересборка Pipfile.lock под 3.13)

### Обоснование
Python 3.7 снят с поддержки 2023-06-27 (EOL). Установка на macOS
нецелесообразна. Ключевые зависимости проекта (dash, pandas,
statsmodels, dash-bootstrap-components, gunicorn) имеют готовые
wheels под Python 3.13, поэтому обновление не требует сборки
из исходников.

### Проверка
- `pipenv lock` — успешно
- `pipenv sync --dev` — успешно
- `pipenv run python -VV` — Python 3.13.15

## Запуск
<заполним после запуска run.py>

## Вывод
Исходное состояние воспроизведено с одним задокументированным
минимальным изменением (обновление требуемой версии Python).
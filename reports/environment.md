# Окружение и запуск

- ОС: Ubuntu
- Python: 3.12
- Виртуальное окружение: .venv
- Ключевые библиотеки: dash, dash-bootstrap-components, pandas, statsmodels, gunicorn
- Команда запуска: python run.py
- Адрес приложения: http://127.0.0.1:8050
- Результат: приложение запускается, главная и страница прогнозирования открываются

## Внесённые изменения для совместимости

1. `light=True` → `color="light"` в `dbc.NavbarSimple` (см. reports/errors.md)
2. `app.run_server(debug=True)` → `app.run(debug=True)` (см. reports/errors.md)

## Исходный коммит

1f65547 — Update README.md (последний коммит автора до моих изменений)

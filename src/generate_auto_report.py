"""
Автоматический HTML-отчёт о качестве данных FishGrow.

Запуск:
    python src/generate_auto_report.py

Генерирует reports/lab01-auto-report.html — самодостаточный файл
(графики встроены как base64).
"""
from __future__ import annotations

import base64
import io
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from data_schema import get_schema  # noqa: E402
from data_schema_validate import validate_dataframe  # noqa: E402


DATA_PATH = ROOT / "assets" / "data" / "Fish.csv"
OUT_PATH = ROOT / "reports" / "lab01-auto-report.html"
FIG_DIR = ROOT / "reports" / "figures"


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_summary_table(df: pd.DataFrame) -> str:
    """HTML-таблица с базовой сводкой."""
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=False),
        "min": df.select_dtypes("number").min(),
        "max": df.select_dtypes("number").max(),
    })
    return summary.to_html(classes="summary-table", border=0, na_rep="")


def build_schema_table() -> str:
    rows = []
    for f in get_schema():
        rows.append({
            "Поле": f.name,
            "Тип": f.dtype,
            "Единицы": f.units or "—",
            "Диапазон": (f"[{f.min_value}; {f.max_value}]"
                         if f.min_value is not None or f.max_value is not None
                         else "—"),
            "Допустимые значения": ", ".join(f.allowed_values) if f.allowed_values else "—",
            "Nullable": "да" if f.nullable else "нет",
            "Роль": f.role,
        })
    return pd.DataFrame(rows).to_html(classes="schema-table", border=0, index=False)


def build_validation_table(results: dict) -> str:
    rows = []
    for name, errs in results.items():
        rows.append({
            "Поле": name,
            "Статус": "OK" if not errs else "ОШИБКА",
            "Детали": "; ".join(errs) if errs else "—",
        })
    df = pd.DataFrame(rows)
    # Подсветка
    def color_status(v):
        return "status-ok" if v == "OK" else "status-error"
    html = df.to_html(classes="validation-table", border=0, index=False, escape=False)
    for ok, err in [("OK", "status-ok"), ("ОШИБКА", "status-error")]:
        html = html.replace(f"<td>{ok}</td>", f'<td class="{err}">{ok}</td>')
    return html


def build_distribution_figure(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(df["Weight"], bins=25, color="steelblue", edgecolor="black")
    ax.set_title("Распределение массы (Weight)")
    ax.set_xlabel("Масса, г")
    ax.set_ylabel("Число наблюдений")
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def build_species_figure(df: pd.DataFrame) -> str:
    counts = df["Species"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Численность по видам")
    ax.set_xlabel("Число наблюдений")
    ax.invert_yaxis()
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def build_html(df: pd.DataFrame) -> str:
    schema_html = build_schema_table()
    validation_html = build_validation_table(validate_dataframe(df))
    summary_html = build_summary_table(df)
    fig1_b64 = build_distribution_figure(df)
    fig2_b64 = build_species_figure(df)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>FishGrow — автоматический отчёт о качестве данных</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 1100px; margin: 30px auto; padding: 0 20px; color: #222; }}
  h1 {{ border-bottom: 3px solid #333; padding-bottom: 8px; }}
  h2 {{ margin-top: 32px; color: #333; }}
  .meta {{ background: #f5f5f5; padding: 12px 18px; border-left: 4px solid #666; margin: 15px 0; }}
  .meta p {{ margin: 4px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: 600; }}
  .status-ok {{ background: #d4edda; color: #155724; font-weight: bold; text-align: center; }}
  .status-error {{ background: #f8d7da; color: #721c24; font-weight: bold; text-align: center; }}
  figure {{ margin: 20px 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid #ddd; }}
</style>
</head>
<body>

<h1>FishGrow — автоматический отчёт о качестве данных</h1>

<div class="meta">
  <p><b>Сгенерировано:</b> {generated}</p>
  <p><b>Датасет:</b> <code>assets/data/Fish.csv</code></p>
  <p><b>Строк / столбцов:</b> {df.shape[0]} / {df.shape[1]}</p>
  <p><b>Лабораторная работа №1, ПетрГУ, 2026</b></p>
</div>

<h2>1. Формальная схема данных</h2>
{schema_html}

<h2>2. Результат проверки по схеме</h2>
{validation_html}

<h2>3. Сводка по столбцам</h2>
{summary_html}

<h2>4. Визуализация</h2>

<figure>
  <img src="data:image/png;base64,{fig1_b64}" alt="Распределение массы">
  <figcaption>Распределение массы рыбы (Weight), г.</figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{fig2_b64}" alt="Численность по видам">
  <figcaption>Численность наблюдений по видам рыб.</figcaption>
</figure>

<h2>5. Что дальше</h2>
<p>Отчёт сгенерирован автоматически по формальной схеме.
При изменении единиц или диапазона ключевых полей валидация и тесты
<code>tests/test_schema.py</code> сработают и подсветят проблему.</p>

</body>
</html>
"""


def main():
    print(f"Загружаю {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print("Строю HTML…")
    html = build_html(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")

    print(f"Сохранено: {OUT_PATH.resolve()}")
    print(f"Размер: {OUT_PATH.stat().st_size / 1024:.1f} КБ")


if __name__ == "__main__":
    main()

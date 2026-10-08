"""
Автоматический HTML-отчёт по датасету.
Собирает:
- сводку (shape, dtypes, пропуски);
- результаты run_schema_guard;
- результаты audit_leakage;
- графики: распределение target, box по видам, корреляции, пропуски;
- описательные статистики.
Возвращает путь к сохранённому HTML.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from src.schema_tests import run_schema_guard
from src.data_checks import (
    run_all_checks, check_duplicates, check_near_duplicates,
    check_unique_leakage_columns,
)

pio.templates.default = "plotly_white"


def _fig_distribution(df: pd.DataFrame, target: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Исходная шкала", "Логарифм"))
    fig.add_trace(go.Histogram(x=df[target], nbinsx=30,
                               marker_color="#4C78A8", name=target),
                  row=1, col=1)
    pos = df.loc[df[target] > 0, target]
    fig.add_trace(go.Histogram(x=np.log(pos), nbinsx=30,
                               marker_color="#F58518", name=f"log({target})"),
                  row=1, col=2)
    fig.update_xaxes(title_text=f"{target}, г", row=1, col=1)
    fig.update_xaxes(title_text=f"ln({target}, г)", row=1, col=2)
    fig.update_yaxes(title_text="Количество, шт.", row=1, col=1)
    fig.update_yaxes(title_text="Количество, шт.", row=1, col=2)
    fig.update_layout(title="Распределение целевой переменной",
                      showlegend=False, height=400)
    return fig


def _fig_by_group(df: pd.DataFrame, group: str, target: str) -> go.Figure:
    counts = df[group].value_counts().sort_values(ascending=False)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Численность по видам",
                                        f"{target} по видам"))
    fig.add_trace(go.Bar(x=counts.index, y=counts.values,
                         marker_color="#54A24B", name="count"),
                  row=1, col=1)
    fig.add_trace(go.Box(x=df[group], y=df[target],
                         marker_color="#E45756", name=target),
                  row=1, col=2)
    fig.update_yaxes(title_text="Количество, шт.", row=1, col=1)
    fig.update_yaxes(title_text=f"{target}, г", row=1, col=2)
    fig.update_layout(title="Численность и распределение по видам",
                      showlegend=False, height=400)
    return fig


def _fig_corr(df: pd.DataFrame, numeric: list[str]) -> go.Figure:
    corr = df[numeric].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="Корреляционная матрица (r Пирсона)", height=500)
    return fig


def _fig_missing(df: pd.DataFrame) -> go.Figure:
    miss = df.isna().sum()
    fig = go.Figure(go.Bar(x=miss.index, y=miss.values,
                           marker_color="#4C78A8"))
    fig.update_yaxes(title_text="Пропусков, шт.")
    fig.update_xaxes(title_text="Столбец")
    fig.update_layout(title="Пропуски по колонкам", height=400)
    return fig


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(classes="table table-sm", border=0, index=False)


def build_html_report(
    df: pd.DataFrame,
    target: str = "Weight",
    group: str = "Species",
    numeric: list[str] | None = None,
    output_path: str | Path = "reports/data_report.html",
) -> Path:
    """Собирает HTML-отчёт и сохраняет его на диск."""
    numeric = numeric or [target, "Length1", "Length2", "Length3",
                          "Height", "Width"]

    # ---- 1. Текстовые отчёты ---------------------------------------------
    checks = run_all_checks(df, target=target, group=group, verbose=False)
    schema = run_schema_guard(df, verbose=False)
    dups = check_duplicates(df)
    near = check_near_duplicates(df, numeric_cols=numeric)
    leak = check_unique_leakage_columns(df, target=target, threshold=0.95)

    # ---- 2. Таблицы ------------------------------------------------------
    dtypes_tbl = checks["dtypes"]
    missing_tbl = checks["missing_infinite"]
    impossible_tbl = checks["impossible_values"]
    leak_tbl = leak.copy()
    schema_tbl = pd.DataFrame(schema["units"] + schema["ranges"])

    # ---- 3. Графики ------------------------------------------------------
    fig_dist = _fig_distribution(df, target)
    fig_group = _fig_by_group(df, group, target)
    fig_corr = _fig_corr(df, numeric)
    fig_missing = _fig_missing(df)

    html_figs = "\n".join([
        pio.to_html(fig, include_plotlyjs=False, full_html=False)
        for fig in (fig_dist, fig_group, fig_corr, fig_missing)
    ])

    # ---- 4. Итоговый HTML ------------------------------------------------
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_schema = "OK" if schema["all_ok"] else "FAIL"
    status_color = "#2E7D32" if schema["all_ok"] else "#C62828"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Отчёт по качеству данных — Fish.csv</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 24px; color: #222; }}
  h1, h2, h3 {{ color: #1A237E; }}
  .status {{ display: inline-block; padding: 4px 10px; border-radius: 6px;
             color: white; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .card {{ border: 1px solid #e0e0e0; border-radius: 8px;
           padding: 12px 16px; margin: 12px 0; }}
  table.table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.table th, table.table td {{ padding: 4px 8px;
      border-bottom: 1px solid #eee; text-align: left; }}
  code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Отчёт по качеству данных: Fish.csv</h1>
<p>Сгенерировано: <code>{generated}</code></p>

<div class="card">
  <h2>Схема данных <span class="status" style="background:{status_color}">{status_schema}</span></h2>
  <p>Формальная проверка единиц, диапазонов и категорий.</p>
  {_table_html(schema_tbl)}
</div>

<div class="card">
  <h2>Общая сводка</h2>
  <ul>
    <li>Строк: <b>{checks["shape"]["rows"]}</b>,
        столбцов: <b>{checks["shape"]["columns"]}</b></li>
    <li>Полных дубликатов строк: <b>{dups["n_duplicated_rows"]}</b></li>
    <li>Почти-дубликатов: <b>{len(near)}</b></li>
    <li>Потенциальных утечек (|corr| ≥ 0.95): 
        <b>{int(leak["maybe_leakage"].sum())}</b></li>
  </ul>
</div>

<div class="card">
  <h2>Типы данных</h2>
  {_table_html(dtypes_tbl)}
</div>

<div class="card">
  <h2>Пропуски и бесконечности</h2>
  {_table_html(missing_tbl)}
</div>

<div class="card">
  <h2>Невозможные значения</h2>
  {_table_html(impossible_tbl)}
</div>

<div class="card">
  <h2>Потенциальные утечки и идентификаторы</h2>
  {_table_html(leak_tbl)}
</div>

<div class="card">
  <h2>Графики</h2>
  {html_figs}
</div>

<div class="card">
  <h2>Геометрическая согласованность</h2>
  <pre>{schema["ranges"]!r}</pre>
</div>

<hr>
<p style="color:#777; font-size: 12px;">
  Отчёт сгенерирован автоматически скриптом <code>src/html_report.py</code>.
  Полный список проверок — в <code>src/data_checks.py</code> и
  <code>src/schema_tests.py</code>.
</p>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
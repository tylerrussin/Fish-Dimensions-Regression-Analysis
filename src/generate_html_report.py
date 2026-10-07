"""Build a self-contained HTML summary of the FishGrow EDA checks."""

from html import escape
from pathlib import Path

import pandas as pd

from src import data_checks as checks


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'assets' / 'data' / 'Fish.csv'
OUTPUT_PATH = ROOT / 'reports' / 'eda_report.html'


def _table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes='data-table', escape=True)


def generate_report(data_path: Path = DATA_PATH, output_path: Path = OUTPUT_PATH) -> Path:
    df = checks.load_data(data_path)
    quality = checks.full_quality_report(df)
    species = checks.check_categories(df).reset_index()
    species.columns = ['Species', 'count', 'share', 'is_rare']

    iqr_mask, iqr = checks.outliers_iqr(df['Weight'])
    mad_scores, median, mad = checks.mad_zscore(df['Weight'])
    mad_mask = mad_scores.abs() > 3.5
    anomaly_mask = iqr_mask | mad_mask | (df['Weight'] <= 0)
    anomalies = df.loc[anomaly_mask].copy()
    anomalies.insert(0, 'source_index', anomalies.index)
    anomalies['iqr_flag'] = iqr_mask.loc[anomalies.index].astype(int).to_numpy()
    anomalies['mad_flag'] = mad_mask.loc[anomalies.index].astype(int).to_numpy()
    anomalies['physical_error'] = (anomalies['Weight'] <= 0).astype(int)

    schema_errors = checks.validate_fish_schema(df)
    figures = ''.join(
        f'<li><a href="figures/{escape(name)}">{escape(name)}</a></li>'
        for name in (
            '01_weight_dist.html', '02_weight_by_species.html', '03_pairplot.html',
            '04_corr.html', '05_quality.html', '06_outliers.html', '07_volume_weight.html',
        )
    )
    validation = 'Схема пройдена' if not schema_errors else '; '.join(schema_errors)
    html = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FishGrow: автоматический EDA-отчёт</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#202a2e;background:#f7f8f6}}
h1,h2{{color:#174a45}}h1{{border-bottom:3px solid #ef9b45;padding-bottom:12px}}
section{{margin:30px 0}}.data-table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}
.data-table th,.data-table td{{border-bottom:1px solid #d9e0dc;padding:8px;text-align:left}}
.data-table th{{background:#e8efeb}}a{{color:#12665d}}.status{{padding:12px;background:#e8efeb;border-left:4px solid #287c68}}
</style></head><body>
<h1>FishGrow: автоматический EDA-отчёт</h1>
<p>Источник: <code>assets/data/Fish.csv</code>. Строк: {len(df)}; столбцов: {len(df.columns)}.</p>
<p class="status">Проверка формальной схемы: {escape(validation)}.</p>
<section><h2>Качество полей</h2>{_table(quality)}</section>
<section><h2>Виды</h2>{_table(species)}</section>
<section><h2>Аномалии массы</h2>
<p>IQR: [{iqr['lower']:.1f}, {iqr['upper']:.1f}] г. MAD: медиана {median:.1f} г, MAD {mad:.1f} г. Порог |zMAD| &gt; 3.5.</p>
{_table(anomalies[['source_index', 'Species', 'Weight', 'iqr_flag', 'mad_flag', 'physical_error']])}</section>
<section><h2>Интерактивные графики</h2><ul>{figures}</ul></section>
</body></html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    return output_path


if __name__ == '__main__':
    print(f'HTML report written to {generate_report()}')
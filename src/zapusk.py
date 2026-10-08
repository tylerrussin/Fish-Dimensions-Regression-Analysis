import pandas as pd
from src.html_report import build_html_report
df = pd.read_csv('../assets/data/Fish.csv', encoding='utf-8-sig')
build_html_report(df)
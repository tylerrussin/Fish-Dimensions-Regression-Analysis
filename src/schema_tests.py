"""
Тест-страж на изменение единиц или диапазона ключевых полей.

Вызывается:
- из ноутбука (assert-стиль);
- из pytest (tests/test_schema_stability.py);
- из HTML-отчёта (в виде секции).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.schema import FISH_SCHEMA, FieldRule, schema_by_name


def check_field_units(
    df: pd.DataFrame,
    rule: FieldRule,
    scale_ratio_threshold: float = 50.0,
) -> dict:
    """
    Проверка единиц измерения для числового поля.

    Идея: медиана поля должна быть «разумной» величиной того же порядка,
    что и ожидаемый диапазон. Если фактическая медиана отличается от центра
    ожидаемого диапазона в scale_ratio_threshold раз — единицы поменялись.
    """
    if rule.kind != "numeric" or rule.name not in df.columns:
        return {"field": rule.name, "skipped": True}

    s = pd.to_numeric(df[rule.name], errors="coerce").dropna()
    if s.empty:
        return {"field": rule.name, "ok": False, "reason": "нет данных"}

    expected_center = (rule.min_value + rule.max_value) / 2
    actual_median = float(s.median())
    if expected_center <= 0:
        ratio = np.nan
    else:
        ratio = max(actual_median / expected_center,
                    expected_center / max(actual_median, 1e-9))

    units_suspicious = (
        np.isfinite(ratio) and ratio > scale_ratio_threshold
    )
    return {
        "field": rule.name,
        "units": rule.units,
        "expected_center": round(expected_center, 3),
        "actual_median": round(actual_median, 3),
        "scale_ratio": round(float(ratio), 3),
        "units_suspicious": bool(units_suspicious),
        "ok": not units_suspicious,
    }


def check_field_range(
    df: pd.DataFrame,
    rule: FieldRule,
) -> dict:
    """
    Проверка диапазона с учётом допуска.

    Фактические min/max могут слегка выходить за границы на величину
    tolerance — это норма. Выход за (граница ± tolerance) — тревога.
    """
    if rule.kind != "numeric" or rule.name not in df.columns:
        return {"field": rule.name, "skipped": True}

    s = pd.to_numeric(df[rule.name], errors="coerce").dropna()
    if s.empty:
        return {"field": rule.name, "ok": False, "reason": "нет данных"}

    lo = rule.min_value - rule.tolerance
    hi = rule.max_value + rule.tolerance
    n_below = int((s < lo).sum())
    n_above = int((s > hi).sum())
    return {
        "field": rule.name,
        "units": rule.units,
        "expected_range": (rule.min_value, rule.max_value),
        "tolerance": rule.tolerance,
        "actual_min": float(s.min()),
        "actual_max": float(s.max()),
        "n_below": n_below,
        "n_above": n_above,
        "range_suspicious": n_below + n_above > 0,
        "ok": n_below + n_above == 0,
    }


def check_categories(df: pd.DataFrame, rule: FieldRule) -> dict:
    """Проверка категориального поля на неизвестные значения."""
    if rule.kind != "categorical" or rule.name not in df.columns:
        return {"field": rule.name, "skipped": True}
    actual = set(df[rule.name].dropna().unique())
    expected = set(rule.allowed_categories)
    unknown = actual - expected
    missing = expected - actual
    return {
        "field": rule.name,
        "unknown_categories": sorted(unknown),
        "missing_categories": sorted(missing),
        "ok": not unknown,
    }


def run_schema_guard(
    df: pd.DataFrame,
    verbose: bool = True,
    raise_on_fail: bool = False,
) -> dict:
    """
    Полный прогон схемы. Возвращает отчёт.
    При raise_on_fail=True — бросает AssertionError, если что-то не так.
    Это удобно для CI и для pytest.
    """
    units_reports, range_reports, cat_reports = [], [], []

    for rule in FISH_SCHEMA:
        if rule.kind == "numeric":
            units_reports.append(check_field_units(df, rule))
            range_reports.append(check_field_range(df, rule))
        elif rule.kind == "categorical":
            cat_reports.append(check_categories(df, rule))

    all_ok = (
        all(r.get("ok", True) for r in units_reports) and
        all(r.get("ok", True) for r in range_reports) and
        all(r.get("ok", True) for r in cat_reports)
    )

    report = {
        "units": units_reports,
        "ranges": range_reports,
        "categories": cat_reports,
        "all_ok": all_ok,
    }

    if verbose:
        print("=== Проверка единиц ===")
        for r in units_reports:
            if r.get("skipped"):
                continue
            flag = "OK" if r["ok"] else "!!"
            print(f"  [{flag}] {r['field']:10s} "
                  f"units={r.get('units')} "
                  f"median={r.get('actual_median')} "
                  f"scale_ratio={r.get('scale_ratio')}")
        print("\n=== Проверка диапазонов ===")
        for r in range_reports:
            if r.get("skipped"):
                continue
            flag = "OK" if r["ok"] else "!!"
            print(f"  [{flag}] {r['field']:10s} "
                  f"expected={r.get('expected_range')} ± {r.get('tolerance')} "
                  f"actual=({r.get('actual_min')}, {r.get('actual_max')}) "
                  f"out_of_range={r.get('n_below', 0) + r.get('n_above', 0)}")
        print("\n=== Проверка категорий ===")
        for r in cat_reports:
            if r.get("skipped"):
                continue
            flag = "OK" if r["ok"] else "!!"
            print(f"  [{flag}] {r['field']:10s} "
                  f"unknown={r.get('unknown_categories')} "
                  f"missing={r.get('missing_categories')}")

    if raise_on_fail and not all_ok:
        raise AssertionError("Схема данных нарушена. См. отчёт выше.")

    return report
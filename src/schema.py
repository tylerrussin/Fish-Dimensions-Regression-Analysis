"""
Формальная схема данных для датасета Fish.csv.

Схема описывает:
- имя поля;
- тип (numeric / categorical);
- роль (target / feature / group / id);
- ожидаемые единицы измерения (для числовых);
- допустимый диапазон значений;
- допустимость пропусков.

Схема используется:
- в data_checks для валидации;
- в html_report для отчёта;
- в tests/test_schema_stability для проверки, что данные не «поехали».
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FieldRule:
    name: str
    kind: str                       # "numeric" | "categorical"
    role: str                       # "target" | "feature" | "group" | "id"
    units: Optional[str] = None     # "г", "см", None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_missing: bool = False
    allowed_categories: tuple[str, ...] = field(default_factory=tuple)
    tolerance: float = 0.0          # допуск на границы диапазона (для теста)


# --- Схема Fish.csv ---------------------------------------------------------

FISH_SCHEMA: tuple[FieldRule, ...] = (
    FieldRule(
        name="Species", kind="categorical", role="group",
        allow_missing=False,
        allowed_categories=("Bream", "Roach", "Whitefish", "Parkki",
                            "Perch", "Pike", "Smelt"),
    ),
    FieldRule(
        name="Weight", kind="numeric", role="target",
        units="г", min_value=0.0, max_value=2000.0,
        allow_missing=False, tolerance=100.0,   # допуск на верхнюю границу
    ),
    FieldRule(
        name="Length1", kind="numeric", role="feature",
        units="см", min_value=0.0, max_value=100.0,
        allow_missing=False, tolerance=5.0,
    ),
    FieldRule(
        name="Length2", kind="numeric", role="feature",
        units="см", min_value=0.0, max_value=100.0,
        allow_missing=False, tolerance=5.0,
    ),
    FieldRule(
        name="Length3", kind="numeric", role="feature",
        units="см", min_value=0.0, max_value=100.0,
        allow_missing=False, tolerance=5.0,
    ),
    FieldRule(
        name="Height", kind="numeric", role="feature",
        units="см", min_value=0.0, max_value=30.0,
        allow_missing=False, tolerance=3.0,
    ),
    FieldRule(
        name="Width", kind="numeric", role="feature",
        units="см", min_value=0.0, max_value=15.0,
        allow_missing=False, tolerance=1.5,
    ),
)


def schema_by_name() -> dict[str, FieldRule]:
    return {r.name: r for r in FISH_SCHEMA}
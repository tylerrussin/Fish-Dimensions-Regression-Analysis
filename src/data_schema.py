"""
Формальная схема данных FishGrow.

Описывает каждое поле датасета: имя, тип, единицы, диапазон,
допустимость пропусков. Используется в:
- src/data_schema_validate.py — проверка датасета на соответствие схеме;
- tests/test_schema.py        — тест, срабатывающий при нарушении схемы;
- src/generate_auto_report.py — автоматический HTML-отчёт.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FieldSchema:
    name: str
    dtype: str            # 'numeric' | 'categorical'
    units: Optional[str]  # 'г', 'см', None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list] = None
    nullable: bool = False
    role: str = "feature"  # 'feature' | 'target' | 'group'

    def describe(self) -> str:
        parts = [self.name, self.dtype]
        if self.units:
            parts.append(f"[{self.units}]")
        if self.min_value is not None or self.max_value is not None:
            parts.append(f"range [{self.min_value}; {self.max_value}]")
        if self.allowed_values:
            parts.append(f"allowed={self.allowed_values}")
        return " ".join(parts)


# Схема определена на основе §3 отчёта и подтверждена README + кодом приложения
FISH_SCHEMA: list[FieldSchema] = [
    FieldSchema(
        name="Species", dtype="categorical", units=None,
        allowed_values=["Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"],
        nullable=False, role="group",
    ),
    FieldSchema(
        name="Weight", dtype="numeric", units="g",
        min_value=0.0, max_value=3000.0,  # физически возможный диапазон
        nullable=False, role="target",
    ),
    FieldSchema(
        name="Length1", dtype="numeric", units="cm",
        min_value=0.0, max_value=200.0, nullable=False,
    ),
    FieldSchema(
        name="Length2", dtype="numeric", units="cm",
        min_value=0.0, max_value=200.0, nullable=False,
    ),
    FieldSchema(
        name="Length3", dtype="numeric", units="cm",
        min_value=0.0, max_value=200.0, nullable=False,
    ),
    FieldSchema(
        name="Height", dtype="numeric", units="cm",
        min_value=0.0, max_value=100.0, nullable=False,
    ),
    FieldSchema(
        name="Width", dtype="numeric", units="cm",
        min_value=0.0, max_value=100.0, nullable=False,
    ),
]


def get_schema() -> list[FieldSchema]:
    return FISH_SCHEMA


def get_schema_dict() -> dict:
    """Возвращает схему как словарь {field_name: schema}."""
    return {f.name: f for f in FISH_SCHEMA}

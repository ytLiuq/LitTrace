from __future__ import annotations


UNIT_NORMALIZATION = {
    ("conductivity", "S/cm"): ("S/m", 100.0),
    ("conductivity", "S cm-1"): ("S/m", 100.0),
    ("conductivity", "mS/cm"): ("S/m", 0.1),
    ("response time", "s"): ("ms", 1000.0),
    ("recovery time", "s"): ("ms", 1000.0),
    ("young's modulus", "GPa"): ("MPa", 1000.0),
    ("tensile strength", "GPa"): ("MPa", 1000.0),
    ("capacity", "mAh g-1"): ("mAh/g", 1.0),
}


def normalize_metric_unit(metric: str, value: float | str, unit: str | None) -> tuple[float | str, str | None, str | None]:
    if unit is None or not isinstance(value, int | float):
        return value, unit, None
    normalized = UNIT_NORMALIZATION.get((metric, unit))
    if not normalized:
        return value, unit, None
    target_unit, factor = normalized
    return float(value) * factor, target_unit, f"Converted {unit} to {target_unit}."

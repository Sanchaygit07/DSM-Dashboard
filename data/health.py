from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class HealthSummary:
    total_rows: int
    total_plants: int
    date_from: Optional[date]
    date_to: Optional[date]
    expected_blocks: int
    present_blocks: int
    missing_blocks: int
    avc_zero_rows: int
    ppa_missing_rows: int
    all_zero_rows: int


def _to_date(x) -> Optional[date]:
    try:
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None


def summarize_health_from_df(
    df: pd.DataFrame,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    plant_col: str = "Plant",
) -> tuple[HealthSummary, Dict[str, Any]]:
    """
    Compute data-health signals from an already-filtered dataset.

    This intentionally runs AFTER filters (region/plant/QCA/date-range) so the
    health panel reflects exactly what the user selected.
    """
    if df is None or df.empty:
        summary = HealthSummary(
            total_rows=0,
            total_plants=0,
            date_from=None,
            date_to=None,
            expected_blocks=0,
            present_blocks=0,
            missing_blocks=0,
            avc_zero_rows=0,
            ppa_missing_rows=0,
            all_zero_rows=0,
        )
        return summary, {"per_plant": pd.DataFrame()}

    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date

    # Determine selected date window (prefer explicit filters)
    sd = _to_date(start_date) if start_date else None
    ed = _to_date(end_date) if end_date else None
    cov_from = d["date"].min() if "date" in d.columns else None
    cov_to = d["date"].max() if "date" in d.columns else None
    date_from = sd or cov_from
    date_to = ed or cov_to
    if date_from and date_to and date_to < date_from:
        date_to = date_from

    # Plant identity for expected-block computation
    plant_col_use = plant_col if plant_col in d.columns else ("plant_name" if "plant_name" in d.columns else None)
    total_plants = int(pd.Series(d[plant_col_use]).nunique()) if plant_col_use else 1

    # Expected blocks (15-min slots): days * 96 * plants
    expected_blocks = 0
    if date_from and date_to:
        days = int((date_to - date_from).days) + 1
        expected_blocks = max(0, days * 96 * max(1, total_plants))

    # Present unique blocks (per plant/date/time_block)
    present_blocks = 0
    if plant_col_use and {"date", "time_block"}.issubset(set(d.columns)):
        present_blocks = int(d[[plant_col_use, "date", "time_block"]].drop_duplicates().shape[0])

    missing_blocks = max(0, expected_blocks - present_blocks)

    # Signals
    avc = pd.to_numeric(d.get("AvC_MW", d.get("avc")), errors="coerce").fillna(0.0)
    ppa = pd.to_numeric(d.get("PPA", d.get("ppa")), errors="coerce")
    sched = pd.to_numeric(d.get("Scheduled_MW", d.get("forecasted_power")), errors="coerce").fillna(0.0)
    actual = pd.to_numeric(d.get("Actual_MW", d.get("actual_power")), errors="coerce").fillna(0.0)

    avc_zero_rows = int((avc == 0).sum())
    ppa_missing_rows = int((ppa.isna() | (ppa == 0)).sum())
    all_zero_rows = int(((avc == 0) & (sched == 0) & (actual == 0)).sum())

    # Per-plant breakdown (helps triage)
    per_plant = pd.DataFrame()
    try:
        if plant_col_use and {"date", "time_block"}.issubset(set(d.columns)):
            grp = d.groupby(plant_col_use, dropna=False)
            per_plant = grp.agg(
                rows=("date", "size"),
                date_from=("date", "min"),
                date_to=("date", "max"),
                present_blocks=("time_block", "count"),
                avc_zero=("AvC_MW", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0).sum())) if "AvC_MW" in d.columns else ("date", lambda s: 0),
                ppa_missing=("PPA", lambda s: int((pd.to_numeric(s, errors="coerce").isna() | (pd.to_numeric(s, errors="coerce") == 0)).sum())) if "PPA" in d.columns else ("date", lambda s: 0),
            ).reset_index().rename(columns={plant_col_use: "Plant"})
    except Exception:
        per_plant = pd.DataFrame()

    summary = HealthSummary(
        total_rows=int(len(d)),
        total_plants=total_plants,
        date_from=date_from,
        date_to=date_to,
        expected_blocks=int(expected_blocks),
        present_blocks=int(present_blocks),
        missing_blocks=int(missing_blocks),
        avc_zero_rows=avc_zero_rows,
        ppa_missing_rows=ppa_missing_rows,
        all_zero_rows=all_zero_rows,
    )

    return summary, {"per_plant": per_plant}


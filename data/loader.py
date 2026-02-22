from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import duckdb
import pandas as pd


@dataclass(frozen=True)
class MasterQuery:
    regions: list[str]
    start_date: str
    end_date: str
    plants: list[str]
    qcas: Optional[list[str]] = None


def _norm_region(r: str) -> str:
    return str(r).strip().lower()


def _norm_upper(s: str) -> str:
    return str(s).strip().upper()


def connect_master(db_path: str, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=read_only)


def list_regions(db_path: str) -> list[str]:
    """Return distinct regions from master table (upper-cased)."""
    with connect_master(db_path, read_only=True) as con:
        rows = con.execute("SELECT DISTINCT region FROM master WHERE region IS NOT NULL ORDER BY region").fetchall()
    return sorted({_norm_region(r[0]).upper() for r in rows if r and r[0]})


def list_plants(db_path: str, regions: Iterable[str]) -> list[str]:
    regs = sorted({_norm_region(r) for r in (regions or []) if str(r).strip()})
    if not regs:
        return []

    placeholders = ", ".join(["?"] * len(regs))
    sql = f"""
        SELECT DISTINCT plant_name
        FROM master
        WHERE lower(region) IN ({placeholders})
          AND plant_name IS NOT NULL
          AND TRIM(plant_name) <> ''
          AND UPPER(TRIM(plant_name)) <> 'UNKNOWN'
        ORDER BY plant_name
    """
    with connect_master(db_path, read_only=True) as con:
        rows = con.execute(sql, regs).fetchall()
    return [r[0] for r in rows if r and r[0]]


def list_qcas(db_path: str, regions: Iterable[str], plants: Optional[Iterable[str]] = None) -> list[str]:
    regs = sorted({_norm_region(r) for r in (regions or []) if str(r).strip()})
    if not regs:
        return []
    plant_list = sorted({_norm_upper(p) for p in (plants or []) if str(p).strip()})

    params: list[str] = []
    clauses: list[str] = []

    clauses.append(f"lower(region) IN ({', '.join(['?'] * len(regs))})")
    params.extend(regs)

    if plant_list:
        clauses.append(f"UPPER(TRIM(plant_name)) IN ({', '.join(['?'] * len(plant_list))})")
        params.extend(plant_list)

    where = " AND ".join(clauses)
    sql = f"""
        SELECT DISTINCT qca
        FROM master
        WHERE {where}
          AND qca IS NOT NULL
          AND TRIM(qca) <> ''
        ORDER BY qca
    """
    with connect_master(db_path, read_only=True) as con:
        rows = con.execute(sql, params).fetchall()
    return [r[0] for r in rows if r and r[0]]


def load_master_frame(db_path: str, query: MasterQuery) -> pd.DataFrame:
    """
    Load a normalized frame for dashboard computations.

    RPC and STU data share the same master schema (avc, forecasted_power, actual_power, ppa).
    STU rows may have non-null qca; optional qcas filter applies when provided.
    Downstream analytics (bands, dynamic error%, aggregation, exports) treat all regions identically.

    Output columns match the existing dashboard expectations:
    - region, plant_name, date, time_block, from_time, to_time
    - AvC_MW, Scheduled_MW, Actual_MW, PPA
    - qca (plus convenience QCA column)
    """
    regs = sorted({_norm_region(r) for r in (query.regions or []) if str(r).strip()})
    if not regs:
        return pd.DataFrame()

    plant_list = sorted({_norm_upper(p) for p in (query.plants or []) if str(p).strip()})
    qca_list = sorted({str(q).strip() for q in (query.qcas or []) if str(q).strip()}) if query.qcas else []

    params: list[str] = []
    clauses: list[str] = []

    clauses.append(f"lower(region) IN ({', '.join(['?'] * len(regs))})")
    params.extend(regs)

    clauses.append("date >= ? AND date <= ?")
    params.extend([query.start_date, query.end_date])

    if plant_list:
        clauses.append(f"UPPER(TRIM(plant_name)) IN ({', '.join(['?'] * len(plant_list))})")
        params.extend(plant_list)

    with connect_master(db_path, read_only=True) as con:
        cols_df = con.execute("PRAGMA table_info(master)").fetchdf()
        col_names = cols_df["name"].tolist() if hasattr(cols_df, "__iter__") else []
        has_qca_col = "qca" in col_names

    if qca_list and has_qca_col:
        clauses.append(f"qca IN ({', '.join(['?'] * len(qca_list))})")
        params.extend(qca_list)

    where = " AND ".join(clauses)
    select_qca = ", qca" if has_qca_col else ", CAST(NULL AS VARCHAR) AS qca"
    sql = f"""
        SELECT
            region,
            plant_name,
            date,
            time_block,
            from_time,
            to_time,
            avc AS AvC_MW,
            forecasted_power AS Scheduled_MW,
            actual_power AS Actual_MW,
            ppa AS PPA
            {select_qca}
        FROM master
        WHERE {where}
        ORDER BY plant_name, date, time_block
    """
    with connect_master(db_path, read_only=True) as con:
        df = con.execute(sql, params).fetchdf()

    if not df.empty:
        df["Plant"] = df["plant_name"]
        df["block"] = df["time_block"]
        df["date_time"] = pd.to_datetime(df["date"]) + pd.to_timedelta((df["time_block"] - 1) * 15, unit="m")
        df["QCA"] = df.get("qca")

    return df


"""
STU ingestion: load STU raw files into Master DuckDB.

Existing sites  : Motala, Agra, Amguri, 50Hz (Rajpimpri + Kudligi-Solar), Ghatodi
New sites added : Bhainsada, DRES_7, DRES_8, Mirkala, Nerale,
                  Ghughrala, Chelur, Chowdankupe, Manhalli, Kuldigi-Wind
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    base = base_dir or _PROJECT_ROOT
    p = Path(path_str)
    return p if p.is_absolute() else (base / p)


def _block_to_time_range(block: int) -> Tuple[str, str]:
    """Convert 1..96 block into (from_time, to_time) as HH:MM strings."""
    block = int(block)
    block = max(1, min(96, block))
    base = datetime(1900, 1, 1)
    start = base + timedelta(minutes=(block - 1) * 15)
    end   = base + timedelta(minutes=block * 15)
    return start.strftime("%H:%M"), end.strftime("%H:%M")


def parse_time_to_master(value) -> Tuple[Optional[object], Optional[int], Optional[str], Optional[str]]:
    """
    Parse 'Date and Time Block' combined column values into master schema fields.
    Returns (date, time_block, from_time, to_time).

    Handles formats like:
        "01/01/2025 - 00:15 - 00:30"
        "2025-01-01 00:00:00 - 00:15"
        "01-Jan-2025 00:00 - 00:15"
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None, None, None

    text = str(value).strip().strip('"').strip("'")
    if not text:
        return None, None, None, None

    if re.search(r"\s+-\s+", text):
        parts = re.split(r"\s+-\s+", text, maxsplit=1)
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            left_dt = pd.to_datetime(left, errors="coerce", dayfirst=False)
            if pd.isna(left_dt):
                left_dt = pd.to_datetime(left, errors="coerce", dayfirst=True)
            if pd.isna(left_dt):
                return None, None, None, None

            right_dt = pd.to_datetime(right, errors="coerce", dayfirst=False)
            if pd.isna(right_dt):
                right_dt = pd.to_datetime(right, errors="coerce", dayfirst=True)

            from_time = left_dt.strftime("%H:%M")
            to_time   = right_dt.strftime("%H:%M") if not pd.isna(right_dt) else (right[:5] if len(right) >= 5 else right)
            minutes   = int(left_dt.hour) * 60 + int(left_dt.minute)
            block     = (minutes // 15) + 1
            return left_dt.date(), block, from_time, to_time

    dt = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None, None, None, None

    minutes   = int(dt.hour) * 60 + int(dt.minute)
    block     = (minutes // 15) + 1
    from_time = dt.strftime("%H:%M")
    to_time   = (dt + timedelta(minutes=15)).strftime("%H:%M")
    return dt.date(), block, from_time, to_time


def _ensure_master_table(conn: duckdb.DuckDBPyConnection, table: str) -> None:
    """Create master table if needed and ensure qca column exists."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            region          VARCHAR NOT NULL,
            plant_name      VARCHAR NOT NULL,
            date            DATE    NOT NULL,
            time_block      INTEGER NOT NULL,
            from_time       VARCHAR NOT NULL,
            to_time         VARCHAR NOT NULL,
            avc             DOUBLE  NOT NULL,
            forecasted_power DOUBLE NOT NULL,
            actual_power    DOUBLE  NOT NULL,
            ppa             DOUBLE  NOT NULL
        );
    """)
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table}_key "
        f"ON {table}(region, plant_name, date, time_block);"
    )
    cols = conn.execute(f"PRAGMA table_info({table})").fetchdf()
    if "qca" not in cols["name"].tolist():
        conn.execute(f"ALTER TABLE {table} ADD COLUMN qca VARCHAR")


def _duckdb_excel_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    try:
        con.execute("LOAD excel")
    except Exception:
        con.execute("INSTALL excel")
        con.execute("LOAD excel")
    return con


def _list_xlsx_sheets(xlsx_path: Path) -> List[str]:
    """List sheet names from an .xlsx without openpyxl."""
    try:
        with zipfile.ZipFile(xlsx_path, "r") as zf:
            xml_bytes = zf.read("xl/workbook.xml")
        root = ET.fromstring(xml_bytes)
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        return [
            el.attrib.get("name", "")
            for el in root.findall(".//m:sheets/m:sheet", ns)
            if el.attrib.get("name")
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Data-quality helper: recover actual_power from error%
# ---------------------------------------------------------------------------

def _apply_actual_power_recovery(
    df: pd.DataFrame,
    *,
    avc_fixed: float,
    plant_name: str,
    error_pct_col: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    For rows where forecast_power > 0 but actual_power == 0:
      - If error_pct_col is present and non-null → recover:
            actual_power = ((error% × avc) / 100) + forecast_power
      - Else → flag for manual review.

    Returns (fixed_df, list_of_warning_strings).
    """
    warnings: List[str] = []

    mask_zero_actual = (
        (pd.to_numeric(df["actual_power"], errors="coerce").fillna(0.0) == 0.0)
        & (pd.to_numeric(df["forecasted_power"], errors="coerce").fillna(0.0) > 0.0)
    )

    if not mask_zero_actual.any():
        return df, warnings

    if error_pct_col and error_pct_col in df.columns:
        err_vals = pd.to_numeric(df.loc[mask_zero_actual, error_pct_col], errors="coerce")
        can_recover = err_vals.notna()
        # Recover where error% is available
        idx_recover = err_vals[can_recover].index
        if len(idx_recover):
            df.loc[idx_recover, "actual_power"] = (
                (err_vals[can_recover] * avc_fixed / 100.0)
                + pd.to_numeric(df.loc[idx_recover, "forecasted_power"], errors="coerce").fillna(0.0)
            )
            tqdm.write(
                f"  [QC] {plant_name}: recovered actual_power for {len(idx_recover)} block(s) "
                f"using Error% formula."
            )

        # Flag where error% is also missing
        idx_missing = err_vals[~can_recover].index
        if len(idx_missing):
            for _, row in df.loc[idx_missing].iterrows():
                month_label = pd.to_datetime(row.get("date"), errors="coerce")
                month_str   = month_label.strftime("%b %Y") if not pd.isna(month_label) else str(row.get("date"))
                warnings.append(
                    f"  [MISSING actual_power] Plant={plant_name}  "
                    f"Month={month_str}  Block={row.get('time_block', '?')}  "
                    f"Date={row.get('date', '?')}"
                )
    else:
        # No error% column at all — flag everything
        for _, row in df.loc[mask_zero_actual].iterrows():
            month_label = pd.to_datetime(row.get("date"), errors="coerce")
            month_str   = month_label.strftime("%b %Y") if not pd.isna(month_label) else str(row.get("date"))
            warnings.append(
                f"  [MISSING actual_power] Plant={plant_name}  "
                f"Month={month_str}  Block={row.get('time_block', '?')}  "
                f"Date={row.get('date', '?')}"
            )

    return df, warnings


# ---------------------------------------------------------------------------
# EXISTING SITES
# ---------------------------------------------------------------------------

def ingest_motala(source_dir: Path, *, region: str, qca: str, ppa: float) -> pd.DataFrame:
    """Ingest Motala STU data from CSV/Excel."""
    folder = source_dir / "Motala"
    if not folder.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    files = sorted(list(folder.glob("*.csv")) + list(folder.glob("*.xls*")))
    for file in tqdm(files, desc="Motala", unit="file", leave=True):
        try:
            raw = pd.read_csv(file) if file.suffix.lower() == ".csv" else pd.read_excel(file)
        except Exception as e:
            tqdm.write(f"WARNING: Motala read failed: {file.name} -> {e}")
            continue

        raw.columns = raw.columns.astype(str).str.strip()
        if "Time Stamp" not in raw.columns:
            tqdm.write(f"WARNING: Motala missing 'Time Stamp': {file.name}")
            continue

        ts  = pd.to_datetime(raw["Time Stamp"], errors="coerce")
        raw = raw.loc[~ts.isna()].copy()
        if raw.empty:
            continue

        ts      = pd.to_datetime(raw["Time Stamp"], errors="coerce")
        minutes = (ts.dt.hour.astype(int) * 60) + ts.dt.minute.astype(int)
        blocks  = (minutes // 15) + 1

        actual_col   = "Actual [MW]"                  if "Actual [MW]"                  in raw.columns else None
        forecast_col = "Accepted_Schedule_EOD [MW]"   if "Accepted_Schedule_EOD [MW]"   in raw.columns else None
        avc_col      = "Reported AvC"                 if "Reported AvC"                 in raw.columns else None

        out = pd.DataFrame({
            "region":           region,
            "plant_name":       "MOTALA",
            "qca":              qca,
            "date":             ts.dt.date,
            "time_block":       blocks.astype(int),
            "from_time":        ts.dt.strftime("%H:%M"),
            "to_time":          (ts + pd.to_timedelta(15, unit="m")).dt.strftime("%H:%M"),
            "avc":              raw[avc_col]      if avc_col      else 0.0,
            "forecasted_power": raw[forecast_col] if forecast_col else 0.0,
            "actual_power":     raw[actual_col]   if actual_col   else 0.0,
            "ppa":              float(ppa),
        })
        frames.append(out)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ingest_agra(source_dir: Path, *, region: str, qca: str, ppa: float) -> pd.DataFrame:
    """Ingest Agra STU data from Excel Calculation Sheet."""
    folder = source_dir / "Agra"
    if not folder.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    files = sorted(list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))
    for file in tqdm(files, desc="Agra", unit="file", leave=True):
        try:
            df = pd.read_excel(file, sheet_name="Calculation Sheet")
        except Exception as e1:
            try:
                con = _duckdb_excel_conn()
                try:
                    df = con.execute(
                        "SELECT * FROM read_xlsx(?, sheet='Calculation Sheet', all_varchar=true)",
                        [str(file)],
                    ).df()
                finally:
                    con.close()
            except Exception as e2:
                tqdm.write(f"WARNING: Agra read failed: {file.name} -> {e1}; fallback: {e2}")
                continue

        df.columns = df.columns.astype(str).str.strip()
        date_col   = next((c for c in df.columns if c.strip().lower() == "date"), None)
        block_col  = next((c for c in df.columns if "block" in c.lower() and "no" in c.lower()), None)
        from_col   = next((c for c in df.columns if "from" in c.lower() and "time" in c.lower()), None)
        to_col     = next((c for c in df.columns if "to" in c.lower() and "time" in c.lower() and "from" not in c.lower()), None)
        avc_col    = next((c for c in df.columns if "avc" in c.lower() and "data" in c.lower()), None)
        fcast_col  = next((c for c in df.columns if c.strip().lower() == "forecast"), None)
        actual_col = next((c for c in df.columns if "actual" in c.lower() and "generation" in c.lower()), None)

        if any(r is None for r in [date_col, block_col, from_col, to_col, avc_col, fcast_col, actual_col]):
            tqdm.write(f"WARNING: Agra missing required columns in {file.name}")
            continue

        base = df[[date_col, block_col, from_col, to_col, avc_col, fcast_col, actual_col]].copy()
        base = base.dropna(subset=[date_col])
        if base.empty:
            continue

        out = pd.DataFrame({
            "region":           region,
            "plant_name":       "TEQ Green-AGRA",
            "qca":              qca,
            "date":             pd.to_datetime(base[date_col], errors="coerce").dt.date,
            "time_block":       pd.to_numeric(base[block_col], errors="coerce").fillna(1).astype(int).clip(1, 96),
            "from_time":        base[from_col].astype(str).str.strip().str.slice(0, 5),
            "to_time":          base[to_col].astype(str).str.strip().str.slice(0, 5),
            "avc":              pd.to_numeric(base[avc_col],    errors="coerce").fillna(0.0),
            "forecasted_power": pd.to_numeric(base[fcast_col],  errors="coerce").fillna(0.0),
            "actual_power":     pd.to_numeric(base[actual_col], errors="coerce").fillna(0.0),
            "ppa":              float(ppa),
        })
        out = out.dropna(subset=["date"])
        frames.append(out)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ingest_amguri(source_dir: Path, *, region: str, qca: str, ppa: float) -> pd.DataFrame:
    """Ingest Amguri STU data from Excel Calculation Sheet."""
    folder = source_dir / "Amguri"
    if not folder.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    files = sorted(list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))
    for file in tqdm(files, desc="Amguri", unit="file", leave=True):
        try:
            df = pd.read_excel(file, sheet_name="Calculation Sheet")
        except Exception as e1:
            try:
                con = _duckdb_excel_conn()
                try:
                    df = con.execute(
                        "SELECT * FROM read_xlsx(?, sheet='Calculation Sheet', all_varchar=true)",
                        [str(file)],
                    ).df()
                finally:
                    con.close()
            except Exception as e2:
                tqdm.write(f"WARNING: Amguri read failed: {file.name} -> {e1}; fallback: {e2}")
                continue

        df.columns = df.columns.astype(str).str.strip()
        date_col   = next((c for c in df.columns if c.strip().lower() == "date"), None)
        block_col  = next((c for c in df.columns if "block" in c.lower() and "no" in c.lower()), None)
        from_col   = next((c for c in df.columns if "from" in c.lower() and "time" in c.lower()), None)
        to_col     = next((c for c in df.columns if c.strip().lower() == "to time" or c.strip().startswith("TO TIME")), None)
        avc_col    = next((c for c in df.columns if "avc" in c.lower() and "data" in c.lower()), None)
        fcast_col  = next((c for c in df.columns if c.strip().lower() == "forecast"), None)
        actual_col = next(
            (c for c in df.columns if "actual" in c.lower() and "generation" in c.lower()),
            next((c for c in df.columns if c.strip().lower() == "actual"), None),
        )

        if any(r is None for r in [date_col, block_col, from_col, to_col, fcast_col, actual_col]):
            tqdm.write(f"WARNING: Amguri missing required columns in {file.name}")
            continue

        cols = [date_col, block_col, from_col, to_col, fcast_col, actual_col]
        if avc_col:
            cols.insert(4, avc_col)
        base = df[cols].copy().dropna(subset=[date_col])
        if base.empty:
            continue

        avc_vals = pd.to_numeric(base[avc_col], errors="coerce").fillna(0.0) if avc_col else 0.0

        out = pd.DataFrame({
            "region":           region,
            "plant_name":       "TEQ Green-ASSAM",
            "qca":              qca,
            "date":             pd.to_datetime(base[date_col], errors="coerce").dt.date,
            "time_block":       pd.to_numeric(base[block_col], errors="coerce").fillna(1).astype(int).clip(1, 96),
            "from_time":        base[from_col].astype(str).str.strip().str.slice(0, 5),
            "to_time":          base[to_col].astype(str).str.strip().str.slice(0, 5),
            "avc":              avc_vals,
            "forecasted_power": pd.to_numeric(base[fcast_col],  errors="coerce").fillna(0.0),
            "actual_power":     pd.to_numeric(base[actual_col], errors="coerce").fillna(0.0),
            "ppa":              float(ppa),
        })
        out = out.dropna(subset=["date"])
        frames.append(out)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ingest_ghatodi(source_dir: Path, *, region: str, qca: str, ppa: float) -> pd.DataFrame:
    """Ingest Ghatodi from STU Raw/Ghatodi.csv."""
    path = source_dir / "Ghatodi.csv"
    if not path.exists():
        return pd.DataFrame()

    for _ in tqdm([1], desc="Ghatodi", unit="file", leave=True):
        try:
            raw = pd.read_csv(path)
        except Exception as e:
            tqdm.write(f"WARNING: Ghatodi read failed: {e}")
            return pd.DataFrame()

        raw.columns = raw.columns.astype(str).str.strip()
        if "Time Stamp" not in raw.columns:
            tqdm.write("WARNING: Ghatodi missing 'Time Stamp' column")
            return pd.DataFrame()

        ts  = pd.to_datetime(raw["Time Stamp"], errors="coerce")
        raw = raw.loc[~ts.isna()].copy()
        if raw.empty:
            return pd.DataFrame()

        ts      = pd.to_datetime(raw["Time Stamp"], errors="coerce")
        minutes = (ts.dt.hour.astype(int) * 60) + ts.dt.minute.astype(int)
        blocks  = (minutes // 15) + 1

        forecast_col = "Accepted_Schedule_EOD [MW]" if "Accepted_Schedule_EOD [MW]" in raw.columns else None
        avc_col      = "Reported AvC"               if "Reported AvC"               in raw.columns else None
        actual_col   = "Actual [MW]"                if "Actual [MW]"                in raw.columns else None

        return pd.DataFrame({
            "region":           region,
            "plant_name":       "GHATODI",
            "qca":              qca,
            "date":             ts.dt.date,
            "time_block":       blocks.astype(int),
            "from_time":        ts.dt.strftime("%H:%M"),
            "to_time":          (ts + pd.to_timedelta(15, unit="m")).dt.strftime("%H:%M"),
            "avc":              raw[avc_col]      if avc_col      else 0.0,
            "forecasted_power": raw[forecast_col] if forecast_col else 0.0,
            "actual_power":     raw[actual_col]   if actual_col   else 0.0,
            "ppa":              float(ppa),
        })


def _read_50hz_table(con: duckdb.DuckDBPyConnection, *, path: Path, sheet: str) -> pd.DataFrame:
    df = con.execute(
        "SELECT * FROM read_xlsx(?, sheet=?, range='A4:E6000', header=true, all_varchar=true)",
        [str(path), sheet],
    ).df()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _build_50hz_rows(
    df: pd.DataFrame,
    *,
    region: str,
    plant_name: str,
    qca: str,
    ppa: float,
    actual_source_col: str,
) -> pd.DataFrame:
    required = ["Date & Time Block", "Forecasted Schedule (MAL)", "AVC", actual_source_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        tqdm.write(f"WARNING: 50Hz sheet missing columns for {plant_name}: {missing}")
        return pd.DataFrame()

    base   = df[required].copy().dropna(subset=["Date & Time Block"])
    parsed = base["Date & Time Block"].apply(parse_time_to_master)
    out    = pd.DataFrame(parsed.tolist(), columns=["date", "time_block", "from_time", "to_time"])
    out    = out.dropna(subset=["date", "time_block"])

    out["region"]           = region
    out["plant_name"]       = plant_name
    out["qca"]              = qca
    out["avc"]              = base.loc[out.index, "AVC"].values
    out["forecasted_power"] = base.loc[out.index, "Forecasted Schedule (MAL)"].values
    out["actual_power"]     = base.loc[out.index, actual_source_col].values
    out["ppa"]              = float(ppa)
    return out.reset_index(drop=True)


def ingest_50hz(source_dir: Path, *, region: str) -> pd.DataFrame:
    """Ingest 50Hz STU data from Excel (Rajpimpri + Kudligi-Solar)."""
    folder = Path(source_dir) / "50Hz"
    if not folder.exists():
        return pd.DataFrame()

    con = _duckdb_excel_conn()
    try:
        frames: List[pd.DataFrame] = []
        files = sorted(folder.glob("*.xls*"))

        for file in tqdm(files, desc="50Hz", unit="file", leave=True):
            sheet_names = _list_xlsx_sheets(file)
            raj_sheet = next((s for s in sheet_names if "rajpimpri" in s.lower()), None) if sheet_names else "Rajpimpri-Panama(Solar)"
            kud_sheet = next((s for s in sheet_names if "kudligi"   in s.lower()), None) if sheet_names else "Kudligi(HYB)-O2(Solar)"

            if raj_sheet:
                try:
                    df = _read_50hz_table(con, path=file, sheet=raj_sheet)
                    frames.append(_build_50hz_rows(
                        df, region=region, plant_name="PSEGPL-RAJPIMPRI",
                        qca="Manikaran", ppa=5.0, actual_source_col="Green Gen-Meter",
                    ))
                except Exception as e:
                    tqdm.write(f"WARNING: 50Hz read failed: {file.name}::{raj_sheet} -> {e}")

            if kud_sheet:
                try:
                    df = _read_50hz_table(con, path=file, sheet=kud_sheet)
                    frames.append(_build_50hz_rows(
                        df, region=region, plant_name="CEPPL SOLAR KUDLIGI",
                        qca="Manikaran", ppa=3.5, actual_source_col="Green Gen-SCADA",
                    ))
                except Exception as e:
                    tqdm.write(f"WARNING: 50Hz read failed: {file.name}::{kud_sheet} -> {e}")

        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# NEW SITES — Group A: Excel "Calculation Sheet"
# (Date, Block No., From Time, To Time, Forecast, Actual already separated)
# ---------------------------------------------------------------------------

def _ingest_calc_sheet_xlsx(
    source_dir: Path,
    folder_name: str,
    *,
    region: str,
    plant_name: str,
    qca: str,
    ppa: float,
    avc_fixed: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generic ingester for sites that use the 'Calculation Sheet' Excel format:
        Date | BLOCK NO. | FROM TIME | TO TIME | Forecast | Actual | Error | Slab | ...

    avc is a fixed value for these sites (not a column in the file).
    Returns (dataframe, list_of_qc_warnings).
    """
    folder = source_dir / folder_name
    if not folder.exists():
        tqdm.write(f"  [SKIP] Folder not found: {folder}")
        return pd.DataFrame(), []

    all_warnings: List[str] = []
    frames: List[pd.DataFrame] = []
    files = sorted(list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))

    for file in tqdm(files, desc=folder_name, unit="file", leave=True):
        try:
            df = pd.read_excel(file, sheet_name="Calculation Sheet")
        except Exception as e1:
            try:
                con = _duckdb_excel_conn()
                try:
                    df = con.execute(
                        "SELECT * FROM read_xlsx(?, sheet='Calculation Sheet', all_varchar=true)",
                        [str(file)],
                    ).df()
                finally:
                    con.close()
            except Exception as e2:
                tqdm.write(f"WARNING: {folder_name} read failed: {file.name} -> {e1}; fallback: {e2}")
                continue

        df.columns = df.columns.astype(str).str.strip()

        # Column discovery (case-insensitive, flexible)
        date_col   = next((c for c in df.columns if c.strip().lower() == "date"), None)
        block_col  = next((c for c in df.columns if re.search(r"block.*no|block\s*#|time\s*block", c, re.I)), None)
        from_col   = next((c for c in df.columns if re.search(r"from\s*time", c, re.I)), None)
        to_col     = next((c for c in df.columns if re.search(r"to\s*time", c, re.I) and not re.search(r"from", c, re.I)), None)
        fcast_col  = next((c for c in df.columns if c.strip().lower() == "forecast"), None)
        actual_col = next((c for c in df.columns if c.strip().lower() == "actual"), None)
        error_col  = next((c for c in df.columns if re.search(r"error", c, re.I)), None)

        missing = [n for n, v in [("Date", date_col), ("BLOCK NO.", block_col),
                                   ("FROM TIME", from_col), ("TO TIME", to_col),
                                   ("Forecast", fcast_col), ("Actual", actual_col)] if v is None]
        if missing:
            tqdm.write(f"WARNING: {folder_name} missing columns {missing} in {file.name}")
            continue

        base = df[[date_col, block_col, from_col, to_col, fcast_col, actual_col]
                  + ([error_col] if error_col else [])].copy()
        base = base.dropna(subset=[date_col])
        if base.empty:
            continue

        out = pd.DataFrame({
            "region":           region,
            "plant_name":       plant_name,
            "qca":              qca,
            "date":             pd.to_datetime(base[date_col], errors="coerce").dt.date,
            "time_block":       pd.to_numeric(base[block_col], errors="coerce").fillna(1).astype(int).clip(1, 96),
            "from_time":        base[from_col].astype(str).str.strip().str.slice(0, 5),
            "to_time":          base[to_col].astype(str).str.strip().str.slice(0, 5),
            "avc":              float(avc_fixed),
            "forecasted_power": pd.to_numeric(base[fcast_col],  errors="coerce").fillna(0.0),
            "actual_power":     pd.to_numeric(base[actual_col], errors="coerce").fillna(0.0),
            "ppa":              float(ppa),
        })
        if error_col:
            out["_error_pct"] = pd.to_numeric(base[error_col].values, errors="coerce")

        out = out.dropna(subset=["date"])

        # ---- Data-quality check ----
        _epcol = "_error_pct" if "_error_pct" in out.columns else None
        out, warns = _apply_actual_power_recovery(
            out, avc_fixed=avc_fixed, plant_name=plant_name, error_pct_col=_epcol
        )
        all_warnings.extend(warns)

        # Drop helper column before stacking
        out = out.drop(columns=["_error_pct"], errors="ignore")
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result, all_warnings


def ingest_bhainsada(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    return _ingest_calc_sheet_xlsx(
        source_dir, "Bhainsada",
        region=region, plant_name="CSPPL-BHAINSADA", qca="REGENT CLIMATE CONNECT",
        ppa=2.49, avc_fixed=250.0,
    )


def ingest_dres7(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    return _ingest_calc_sheet_xlsx(
        source_dir, "DRES_7",
        region=region, plant_name="DRE7", qca="REGENT CLIMATE CONNECT",
        ppa=6.32, avc_fixed=7.0,
    )


def ingest_dres8(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    """DRES_8 has its own folder: STU Raw/DRES_8/"""
    return _ingest_calc_sheet_xlsx(
        source_dir, "DRES_8",
        region=region, plant_name="DRE8", qca="REGENT CLIMATE CONNECT",
        ppa=6.32, avc_fixed=8.0,
    )


# ---------------------------------------------------------------------------
# NEW SITES — Group B: CSV "Date and Time Block" combined column
# (skip 3 rows, header on row 4, data from row 5)
# ---------------------------------------------------------------------------

def _ingest_dateblock_csv(
    source_dir: Path,
    folder_name: str,
    *,
    region: str,
    plant_name: str,
    qca: str,
    ppa: float,
    avc_fixed: float,
    actual_source_col: str,
    actual_source_col_fallbacks: Optional[List[str]] = None,
    sheet_name: str = "Power Forecast VS Actual Genera",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generic ingester for sites with a combined 'Date and Time Block' column.

    actual_source_col_fallbacks: tried in order if primary col not found.
    skiprows=3 → header row 4, data from row 5.
    """
    folder = source_dir / folder_name
    if not folder.exists():
        tqdm.write(f"  [SKIP] Folder not found: {folder}")
        return pd.DataFrame(), []

    all_warnings: List[str] = []
    frames: List[pd.DataFrame] = []
    files = sorted(
        list(folder.glob("*.csv"))
        + list(folder.glob("*.xlsx"))
        + list(folder.glob("*.xls"))
    )

    for file in tqdm(files, desc=folder_name, unit="file", leave=True):
        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file, skiprows=3, header=0)
            else:
                df = pd.read_excel(file, sheet_name=sheet_name, skiprows=3, header=0)
        except Exception as e:
            tqdm.write(f"WARNING: {folder_name} read failed: {file.name} -> {e}")
            continue

        df.columns = df.columns.astype(str).str.strip()

        dtb_col    = next((c for c in df.columns if re.search(r"date.*time.*block|time.*block.*date", c, re.I)), None)
        fcast_col  = next((c for c in df.columns if re.search(r"forecast.*schedule.*mal|forecasted.*schedule", c, re.I)), None)
        # Resolve actual column: primary name first, then fallbacks
        actual_col = next((c for c in df.columns if c.strip() == actual_source_col), None)
        if actual_col is None and actual_source_col_fallbacks:
            for _fb in actual_source_col_fallbacks:
                actual_col = next((c for c in df.columns if c.strip() == _fb), None)
                if actual_col is not None:
                    tqdm.write(f"  [INFO] {folder_name}: using fallback actual col '{_fb}' in {file.name}")
                    break
        error_col  = next((c for c in df.columns if re.search(r"error.*%|error\s*in\s*%|%.*error", c, re.I)), None)

        if dtb_col is None:
            tqdm.write(f"WARNING: {folder_name} missing 'Date and Time Block' in {file.name} — cols: {list(df.columns)[:8]}")
            continue
        if fcast_col is None:
            tqdm.write(f"WARNING: {folder_name} missing Forecast col in {file.name}")
            continue
        if actual_col is None:
            tried = [actual_source_col] + (actual_source_col_fallbacks or [])
            tqdm.write(f"WARNING: {folder_name} missing actual col (tried: {tried}) in {file.name} — cols: {list(df.columns)}")
            continue

        base = df[[dtb_col, fcast_col, actual_col]
                  + ([error_col] if error_col else [])].copy()
        base = base.dropna(subset=[dtb_col])
        if base.empty:
            continue

        parsed = base[dtb_col].apply(parse_time_to_master)
        time_df = pd.DataFrame(parsed.tolist(), columns=["date", "time_block", "from_time", "to_time"])
        time_df = time_df.dropna(subset=["date", "time_block"])

        out = pd.DataFrame({
            "region":           region,
            "plant_name":       plant_name,
            "qca":              qca,
            "date":             time_df["date"].values,
            "time_block":       pd.to_numeric(time_df["time_block"], errors="coerce").fillna(1).astype(int).clip(1, 96).values,
            "from_time":        time_df["from_time"].values,
            "to_time":          time_df["to_time"].values,
            "avc":              float(avc_fixed),
            "forecasted_power": pd.to_numeric(base.loc[time_df.index, fcast_col],  errors="coerce").fillna(0.0).values,
            "actual_power":     pd.to_numeric(base.loc[time_df.index, actual_col], errors="coerce").fillna(0.0).values,
            "ppa":              float(ppa),
        })
        if error_col:
            out["_error_pct"] = pd.to_numeric(base.loc[time_df.index, error_col].values, errors="coerce")

        out = out.dropna(subset=["date"])

        # ---- Data-quality check ----
        _epcol = "_error_pct" if "_error_pct" in out.columns else None
        out, warns = _apply_actual_power_recovery(
            out, avc_fixed=avc_fixed, plant_name=plant_name, error_pct_col=_epcol
        )
        all_warnings.extend(warns)
        out = out.drop(columns=["_error_pct"], errors="ignore")
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result, all_warnings


def ingest_mirkala(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    return _ingest_dateblock_csv(
        source_dir, "Mirkala",
        region=region, plant_name="PWEGPL-MIRKALA", qca="MANIKARAN",
        ppa=5.73, avc_fixed=80.0, actual_source_col="Green Gen-Meter",
    )


def ingest_nerale(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    return _ingest_dateblock_csv(
        source_dir, "Nerale",
        region=region, plant_name="PWEPL-NERALE", qca="MANIKARAN",
        ppa=5.78, avc_fixed=70.4, actual_source_col="Green Gen-SCADA",
    )


def ingest_chelur(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    return _ingest_dateblock_csv(
        source_dir, "Chelur",
        region=region, plant_name="CHELUR", qca="MANIKARAN",
        ppa=6.1, avc_fixed=17.1, actual_source_col="Green Gen-SCADA",
    )


def ingest_chowdankupe(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    return _ingest_dateblock_csv(
        source_dir, "Chowdankupe",
        region=region, plant_name="Chowdankupe", qca="MANIKARAN",
        ppa=6.1, avc_fixed=9.4, actual_source_col="Green Gen-Meter",
        sheet_name="Monthwise",
    )


def ingest_manhalli(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    return _ingest_dateblock_csv(
        source_dir, "Manhalli",
        region=region, plant_name="Manhalli", qca="MANIKARAN",
        ppa=6.1, avc_fixed=9.57, actual_source_col="Green Gen-Meter",
    )


def ingest_kuldigi_wind(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Kuldigi-Wind actual power column is named 'Active' (confirmed from source file).
    Falls back to 'Green Gen-SCADA' if 'Active' is not found, for forward-compatibility.
    """
    return _ingest_dateblock_csv(
        source_dir, "Kuldigi-Wind",
        region=region, plant_name="CEPPL-WIND-KUDLIGI", qca="MANIKARAN",
        ppa=3.5, avc_fixed=50.6,
        actual_source_col="Active",             # ← confirmed column name from source
        actual_source_col_fallbacks=["Green Gen-SCADA", "Actual"],
    )


# ---------------------------------------------------------------------------
# NEW SITE — Group C: Ghughrala (Excel in STU Raw root, sheet "JAN 26" etc.)
# ---------------------------------------------------------------------------

# Month abbreviation → month number
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_sheet_date(sheet_name: str) -> Optional[datetime]:
    """
    Parse Ghughrala sheet names into (year, month).
    Handles all observed formats:
      "JAN 26"    "JAN 2026"   (space separator)
      "Jan-26"    "Jan-2026"   (dash separator  ← actual file format)
      "January 26"             (full month name)
    Returns a datetime for the 1st of that month/year, or None if unrecognised.
    """
    text = sheet_name.strip().upper()
    # Normalise separator: replace dash or hyphen with space
    text = re.sub(r"[-–—]", " ", text).strip()
    # Pattern: "MMM YY", "MMM YYYY", "MMMMMM YY", etc.
    m = re.match(r"([A-Z]+)\s+(\d{2,4})$", text)
    if m:
        mon_str, yr_str = m.group(1), m.group(2)
        mon = _MONTH_MAP.get(mon_str[:3])
        if mon:
            yr = int(yr_str)
            if yr < 100:
                yr += 2000
            try:
                return datetime(yr, mon, 1)
            except ValueError:
                pass
    return None


def _parse_block_time(val) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Parse Ghughrala 'Block time' column value like "00:00 - 00:15" into
    (from_time, to_time, time_block).

    Handles:
        "00:00 - 00:15"      → ("00:00", "00:15", 1)
        "00:15 - 00:30"      → ("00:15", "00:30", 2)
        "23:45 - 00:00"      → ("23:45", "00:00", 96)
        datetime objects     → extract HH:MM as from_time, add 15 min for to_time
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None, None, None

    # If it's already a datetime/time object (Excel sometimes parses these)
    if hasattr(val, "hour"):
        h, m   = int(val.hour), int(val.minute)
        from_t = f"{h:02d}:{m:02d}"
        end_dt = datetime(1900, 1, 1, h, m) + timedelta(minutes=15)
        to_t   = end_dt.strftime("%H:%M")
        block  = (h * 60 + m) // 15 + 1
        return from_t, to_t, block

    text = str(val).strip()

    # Pattern "HH:MM - HH:MM" or "HH:MM – HH:MM"
    m = re.match(r"(\d{1,2}:\d{2})\s*[-–—]\s*(\d{1,2}:\d{2})", text)
    if m:
        from_t, to_t = m.group(1).strip(), m.group(2).strip()
        # Derive time_block from from_time
        try:
            hh, mm = map(int, from_t.split(":"))
            block  = (hh * 60 + mm) // 15 + 1
        except Exception:
            block = None
        return from_t, to_t, block

    return None, None, None


def ingest_ghughrala(source_dir: Path, *, region: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ingest Ghughrala from a SINGLE fixed file: STU Raw/GHUGHRALA.xlsx
    Each month is a separate sheet named "JAN 26", "Jan-26", "FEB 26", etc.
    Skip 2 rows → header on row 3, data from row 4.

    Column mapping (confirmed from source file):
        Time Block      → date         (contains dates like 01-01-2026)
        Block time      → from_time, to_time  (contains "00:00 - 00:15")
        Schedule        → forecasted_power
        Actual          → actual_power
        Available Cap.  → avc (fixed 70.2 used instead)
        MAE %           → error% for actual_power recovery

    DB fixed values:
        plant_name : TEQXIIPL-Ghughrala
        avc        : 70.2
        ppa        : 5.78
        region     : stu
        qca        : Unilink
    """
    all_warnings: List[str] = []
    frames:       List[pd.DataFrame] = []

    AVC_FIXED = 70.2
    PPA_FIXED = 5.78
    PLANT     = "TEQXIIPL-Ghughrala"
    QCA       = "Unilink"

    # ── Fixed file location: STU Raw/GHUGHRALA.xlsx ──
    ghughrala_file = source_dir / "GHUGHRALA.xlsx"
    if not ghughrala_file.exists():
        # Try case-insensitive match (Windows is case-insensitive but Linux is not)
        matches = list(source_dir.glob("[Gg][Hh][Uu][Gg][Hh][Rr][Aa][Ll][Aa]*.xlsx"))
        if not matches:
            tqdm.write(f"  [SKIP] GHUGHRALA.xlsx not found in {source_dir}")
            return pd.DataFrame(), []
        ghughrala_file = matches[0]

    tqdm.write(f"  Ghughrala file: {ghughrala_file.name}")
    sheet_names = _list_xlsx_sheets(ghughrala_file)
    tqdm.write(f"  Sheets found: {sheet_names}")

    month_sheets = [(s, _parse_sheet_date(s)) for s in sheet_names]
    month_sheets = [(s, d) for s, d in month_sheets if d is not None]

    if not month_sheets:
        tqdm.write(f"  [SKIP] No recognisable month sheets in {ghughrala_file.name}. "
                   f"Found sheets: {sheet_names}")
        return pd.DataFrame(), []

    for sheet, _sheet_date in tqdm(month_sheets, desc="Ghughrala", unit="sheet", leave=True):
        try:
            df = pd.read_excel(ghughrala_file, sheet_name=sheet, skiprows=2, header=0)
        except Exception as e:
            tqdm.write(f"WARNING: Ghughrala read failed: {ghughrala_file.name}::{sheet} -> {e}")
            continue

        df.columns = df.columns.astype(str).str.strip()
        tqdm.write(f"    Sheet '{sheet}' columns: {list(df.columns)}")

        # ── Column discovery ──
        # "Time Block" column  → contains dates (01-01-2026)
        date_col   = next((c for c in df.columns if re.search(r"time\s*block$", c, re.I)), None)
        # "Block time" column  → contains "00:00 - 00:15" time ranges
        btime_col  = next((c for c in df.columns if re.search(r"block\s*time", c, re.I)), None)
        fcast_col  = next((c for c in df.columns if c.strip().lower() == "schedule"), None)
        actual_col = next((c for c in df.columns if c.strip().lower() == "actual"), None)
        error_col  = next((c for c in df.columns if re.search(r"mae\s*%", c, re.I)), None)

        missing = [n for n, v in [("Time Block(date)", date_col),
                                   ("Block time",       btime_col),
                                   ("Schedule",         fcast_col),
                                   ("Actual",           actual_col)] if v is None]
        if missing:
            tqdm.write(f"WARNING: Ghughrala sheet '{sheet}' missing cols: {missing}. "
                       f"Available: {list(df.columns)}")
            continue

        # ── Build working frame ──
        use_cols = [date_col, btime_col, fcast_col, actual_col] + ([error_col] if error_col else [])
        base = df[use_cols].copy()

        # Drop rows where BOTH date and block_time are null
        base = base.dropna(subset=[date_col, btime_col], how="all")
        # Forward-fill date (date only repeats on first block of each day in some files)
        base[date_col] = base[date_col].ffill()
        base = base.dropna(subset=[btime_col])   # block_time must always be present
        if base.empty:
            tqdm.write(f"    Sheet '{sheet}': no data rows after cleaning.")
            continue

        # ── Parse dates ──
        dates = pd.to_datetime(base[date_col], errors="coerce", dayfirst=True).dt.date
        # Fall back to sheet date for any rows that failed
        dates = dates.where(dates.notna(), other=_sheet_date.date())

        # ── Parse Block time → from_time, to_time, time_block ──
        parsed_times = base[btime_col].apply(_parse_block_time)
        from_times   = [t[0] for t in parsed_times]
        to_times     = [t[1] for t in parsed_times]
        time_blocks  = [t[2] for t in parsed_times]

        out = pd.DataFrame({
            "region":           region,
            "plant_name":       PLANT,
            "qca":              QCA,
            "date":             dates.values,
            "time_block":       pd.array(time_blocks, dtype="Int64"),
            "from_time":        from_times,
            "to_time":          to_times,
            "avc":              float(AVC_FIXED),
            "forecasted_power": pd.to_numeric(base[fcast_col].values,  errors="coerce").astype(float),
            "actual_power":     pd.to_numeric(base[actual_col].values, errors="coerce").astype(float),
            "ppa":              float(PPA_FIXED),
        })

        if error_col:
            out["_error_pct"] = pd.to_numeric(base[error_col].values, errors="coerce")

        # Drop rows with unparseable block_time
        out = out.dropna(subset=["time_block", "from_time"])
        out["time_block"] = out["time_block"].astype(int).clip(1, 96)
        out["forecasted_power"] = out["forecasted_power"].fillna(0.0)
        out["actual_power"]     = out["actual_power"].fillna(0.0)

        # ── Data-quality: recover actual_power where forecast > 0 but actual == 0 ──
        _epcol = "_error_pct" if "_error_pct" in out.columns else None
        out, warns = _apply_actual_power_recovery(
            out, avc_fixed=AVC_FIXED, plant_name=PLANT, error_pct_col=_epcol
        )
        all_warnings.extend(warns)
        out = out.drop(columns=["_error_pct"], errors="ignore")

        tqdm.write(f"    Sheet '{sheet}': {len(out)} rows parsed.")
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tqdm.write(f"  Ghughrala total rows: {len(result):,}")
    return result, all_warnings


# ---------------------------------------------------------------------------
# Cleaning & master merge (unchanged from original)
# ---------------------------------------------------------------------------

def _clean_for_master(df: pd.DataFrame, *, region_default: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "region" not in out.columns:
        out["region"] = region_default
    out["region"]     = out["region"].fillna(region_default).astype(str).str.strip().str.lower()
    out["plant_name"] = out.get("plant_name", "UNKNOWN").astype(str).str.strip()
    out["plant_name"] = out["plant_name"].replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})

    if "qca" not in out.columns:
        out["qca"] = None
    out["qca"] = out["qca"].astype(str).replace({"nan": None, "None": None})

    out["date"]       = pd.to_datetime(out["date"], errors="coerce").dt.date
    out              = out.dropna(subset=["date"])
    out["time_block"] = pd.to_numeric(out["time_block"], errors="coerce").fillna(1).astype(int).clip(1, 96)

    if "from_time" not in out.columns:
        out["from_time"] = None
    if "to_time" not in out.columns:
        out["to_time"] = None
    missing_time = (
        out["from_time"].isna()
        | out["to_time"].isna()
        | (out["from_time"].astype(str).str.len() < 4)
    )
    if missing_time.any():
        ft_tt = out.loc[missing_time, "time_block"].apply(_block_to_time_range)
        out.loc[missing_time, "from_time"] = [x[0] for x in ft_tt]
        out.loc[missing_time, "to_time"]   = [x[1] for x in ft_tt]

    out["from_time"] = out["from_time"].astype(str).str.slice(0, 5)
    out["to_time"]   = out["to_time"].astype(str).str.slice(0, 5)

    for col in ["avc", "forecasted_power", "actual_power", "ppa"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out.drop_duplicates(subset=["region", "plant_name", "date", "time_block"], keep="first")
    return out


def merge_into_master(
    df: pd.DataFrame,
    *,
    db_path: Path,
    table: str,
    region_default: str,
    base_dir: Optional[Path] = None,
) -> None:
    if df.empty:
        print("No STU data found.")
        return

    base        = base_dir or _PROJECT_ROOT
    resolved_db = db_path if db_path.is_absolute() else (base / db_path)
    os.makedirs(resolved_db.parent, exist_ok=True)

    db_str = str(resolved_db)
    for attempt in range(3):
        try:
            conn = duckdb.connect(db_str)
            break
        except Exception as e:
            err = str(e).lower()
            if ("cannot open" in err or "in use" in err or "another process" in err) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print("\n[STU] master.duckdb is in use. Stop the dashboard (Ctrl+C) and run again.")
            raise

    try:
        pbar = tqdm(total=1, desc="Merging into master", unit="batch", leave=True)
        _ensure_master_table(conn, table)

        df = _clean_for_master(df, region_default=region_default)
        if df.empty:
            pbar.close()
            print("No usable STU rows after cleaning.")
            return

        cols        = conn.execute(f"PRAGMA table_info({table})").fetchdf()
        master_cols = cols["name"].tolist()

        if "qca" not in master_cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN qca VARCHAR")
            master_cols = conn.execute(f"PRAGMA table_info({table})").fetchdf()["name"].tolist()

        incoming = df.copy()
        for col in master_cols:
            if col not in incoming.columns:
                if col in {"avc", "forecasted_power", "actual_power", "ppa"}:
                    incoming[col] = 0.0
                elif col in {"region", "plant_name", "from_time", "to_time"}:
                    incoming[col] = ""
                else:
                    incoming[col] = None
        incoming = incoming[master_cols]
        incoming = incoming.drop_duplicates(subset=["region", "plant_name", "date", "time_block"], keep="first")

        conn.register("incoming", incoming)
        existing = conn.execute(f"""
            SELECT COUNT(*)
            FROM {table} m
            WHERE EXISTS (
                SELECT 1 FROM incoming i
                WHERE m.region     = i.region
                  AND m.plant_name = i.plant_name
                  AND m.date       = i.date
                  AND m.time_block = i.time_block
            );
        """).fetchone()[0]

        col_list = ", ".join(master_cols)
        conn.execute(f"""
            INSERT INTO {table} ({col_list})
            SELECT {col_list}
            FROM incoming i
            WHERE NOT EXISTS (
                SELECT 1 FROM {table} m
                WHERE m.region     = i.region
                  AND m.plant_name = i.plant_name
                  AND m.date       = i.date
                  AND m.time_block = i.time_block
            )
            ORDER BY i.region, i.plant_name, i.date, i.time_block;
        """)

        inserted = max(0, len(incoming) - int(existing))
        conn.unregister("incoming")
        pbar.update(1)
        pbar.close()

        print(f"Inserted {inserted:,} new rows into {resolved_db.name}::{table}")
        print(f"   Incoming rows (deduped): {len(incoming):,}")
        print(f"   Skipped (already exist): {min(len(incoming), int(existing)):,}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stu_ingestion(
    source_dir: str = "STU Raw",
    db_path: str    = "master.duckdb",
    table: str      = "master",
    region: str     = "stu",
    motala_qca: str     = "Reconnect",
    motala_ppa: float   = 3.48,
    dry_run: bool       = False,
    base_dir: Optional[Path] = None,
) -> None:
    """Run full STU ingestion pipeline (existing + all 10 new sites)."""
    base        = base_dir or _PROJECT_ROOT
    source_path = _coerce_path(source_dir, base)
    db_resolved = _coerce_path(db_path, base)

    print("=" * 60)
    print("Loading STU data...")
    print(f"Source : {source_path}")
    print(f"Target : {db_resolved}::{table}")
    print("=" * 60)

    all_qc_warnings: List[str] = []

    # ---- Existing sites ----
    motala = ingest_motala(source_path, region=region, qca=motala_qca, ppa=motala_ppa)
    hz50   = ingest_50hz(source_path, region=region)
    agra   = ingest_agra(source_path,   region=region, qca="Reconnect", ppa=5.0)
    amguri = ingest_amguri(source_path, region=region, qca="Reconnect", ppa=3.798)
    ghatodi = ingest_ghatodi(source_path, region=region, qca="Reconnect", ppa=3.7)

    # ---- New sites ----
    bhainsada,   w1 = ingest_bhainsada(source_path,   region=region)
    dres7,       w2 = ingest_dres7(source_path,       region=region)
    dres8,       w3 = ingest_dres8(source_path,       region=region)
    mirkala,     w4 = ingest_mirkala(source_path,     region=region)
    nerale,      w5 = ingest_nerale(source_path,      region=region)
    ghughrala,   w6 = ingest_ghughrala(source_path,   region=region)
    chelur,      w7 = ingest_chelur(source_path,      region=region)
    chowdankupe, w8 = ingest_chowdankupe(source_path, region=region)
    manhalli,    w9 = ingest_manhalli(source_path,    region=region)
    kuldigi_wind,w10 = ingest_kuldigi_wind(source_path, region=region)

    for w in (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10):
        all_qc_warnings.extend(w)

    # ---- Combine ----
    frames: List[pd.DataFrame] = []
    for d in (motala, hz50, agra, amguri, ghatodi,
              bhainsada, dres7, dres8, mirkala, nerale,
              ghughrala, chelur, chowdankupe, manhalli, kuldigi_wind):
        if d is not None and not d.empty:
            frames.append(d)

    final_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print()
    print("=" * 60)
    print("Row counts per site:")
    print(f"  Motala        : {len(motala):>8,}")
    print(f"  50Hz          : {len(hz50):>8,}")
    print(f"  Agra          : {len(agra):>8,}")
    print(f"  Amguri        : {len(amguri):>8,}")
    print(f"  Ghatodi       : {len(ghatodi):>8,}")
    print(f"  Bhainsada     : {len(bhainsada):>8,}")
    print(f"  DRES_7        : {len(dres7):>8,}")
    print(f"  DRES_8        : {len(dres8):>8,}")
    print(f"  Mirkala       : {len(mirkala):>8,}")
    print(f"  Nerale        : {len(nerale):>8,}")
    print(f"  Ghughrala     : {len(ghughrala):>8,}")
    print(f"  Chelur        : {len(chelur):>8,}")
    print(f"  Chowdankupe   : {len(chowdankupe):>8,}")
    print(f"  Manhalli      : {len(manhalli):>8,}")
    print(f"  Kuldigi-Wind  : {len(kuldigi_wind):>8,}")
    print(f"  ─────────────────────────")
    print(f"  TOTAL         : {len(final_df):>8,}")
    print("=" * 60)

    # ---- QC warnings summary ----
    if all_qc_warnings:
        print()
        print("⚠️  DATA QUALITY WARNINGS — Missing / recovered actual_power blocks:")
        print("   (Please arrange source data for the following and re-run)")
        print()
        for w in all_qc_warnings:
            print(w)
        print()
        print(f"  Total flagged blocks: {len(all_qc_warnings)}")
        print("=" * 60)
    else:
        print("✅  No data quality issues found.")

    if dry_run:
        cleaned = _clean_for_master(final_df, region_default=region)
        print(f"Dry-run: rows after cleaning/dedup: {len(cleaned):,}")
        return

    merge_into_master(
        final_df,
        db_path=db_resolved,
        table=table,
        region_default=region,
        base_dir=base,
    )
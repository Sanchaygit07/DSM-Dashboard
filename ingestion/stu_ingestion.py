"""
STU ingestion: load STU raw files into Master DuckDB.
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


def _coerce_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    base = base_dir or _PROJECT_ROOT
    p = Path(path_str)
    return p if p.is_absolute() else (base / p)


def _block_to_time_range(block: int) -> Tuple[str, str]:
    """Convert 1..96 block into (from_time, to_time) as HH:MM strings."""
    block = int(block)
    block = 1 if block < 1 else (96 if block > 96 else block)
    base = datetime(1900, 1, 1)
    start = base + timedelta(minutes=(block - 1) * 15)
    end = base + timedelta(minutes=block * 15)
    return start.strftime("%H:%M"), end.strftime("%H:%M")


def parse_time_to_master(value) -> Tuple[Optional[object], Optional[int], Optional[str], Optional[str]]:
    """
    Parse time values found in STU sources into master schema fields.
    Returns (date, time_block, from_time, to_time).
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
            to_time = right_dt.strftime("%H:%M") if not pd.isna(right_dt) else (right[:5] if len(right) >= 5 else right)
            minutes = int(left_dt.hour) * 60 + int(left_dt.minute)
            block = (minutes // 15) + 1
            return left_dt.date(), block, from_time, to_time

    dt = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None, None, None, None

    minutes = int(dt.hour) * 60 + int(dt.minute)
    block = (minutes // 15) + 1
    from_time = dt.strftime("%H:%M")
    to_time = (dt + timedelta(minutes=15)).strftime("%H:%M")
    return dt.date(), block, from_time, to_time


def _ensure_master_table(conn: duckdb.DuckDBPyConnection, table: str) -> None:
    """Create master table if needed and ensure qca column exists."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            region VARCHAR NOT NULL,
            plant_name VARCHAR NOT NULL,
            date DATE NOT NULL,
            time_block INTEGER NOT NULL,
            from_time VARCHAR NOT NULL,
            to_time VARCHAR NOT NULL,
            avc DOUBLE NOT NULL,
            forecasted_power DOUBLE NOT NULL,
            actual_power DOUBLE NOT NULL,
            ppa DOUBLE NOT NULL
        );
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_key ON {table}(region, plant_name, date, time_block);")
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
    """List sheet names from an .xlsx without using openpyxl."""
    try:
        with zipfile.ZipFile(xlsx_path, "r") as zf:
            xml_bytes = zf.read("xl/workbook.xml")
        root = ET.fromstring(xml_bytes)
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        return [el.attrib.get("name", "") for el in root.findall(".//m:sheets/m:sheet", ns) if el.attrib.get("name")]
    except Exception:
        return []


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

        ts = pd.to_datetime(raw["Time Stamp"], errors="coerce")
        raw = raw.loc[~ts.isna()].copy()
        if raw.empty:
            continue

        ts = pd.to_datetime(raw["Time Stamp"], errors="coerce")
        minutes = (ts.dt.hour.astype(int) * 60) + ts.dt.minute.astype(int)
        blocks = (minutes // 15) + 1

        from_time = ts.dt.strftime("%H:%M")
        to_time = (ts + pd.to_timedelta(15, unit="m")).dt.strftime("%H:%M")

        actual_col = "Actual [MW]" if "Actual [MW]" in raw.columns else None
        forecast_col = "Accepted_Schedule_EOD [MW]" if "Accepted_Schedule_EOD [MW]" in raw.columns else None
        avc_col = "Reported AvC" if "Reported AvC" in raw.columns else None

        out = pd.DataFrame({
            "region": region,
            "plant_name": "MOTALA",
            "qca": qca,
            "date": ts.dt.date,
            "time_block": blocks.astype(int),
            "from_time": from_time,
            "to_time": to_time,
            "avc": raw[avc_col] if avc_col else 0.0,
            "forecasted_power": raw[forecast_col] if forecast_col else 0.0,
            "actual_power": raw[actual_col] if actual_col else 0.0,
            "ppa": float(ppa),
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
        date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
        block_col = next((c for c in df.columns if "block" in c.lower() and "no" in c.lower()), None)
        from_col = next((c for c in df.columns if "from" in c.lower() and "time" in c.lower()), None)
        to_col = next((c for c in df.columns if "to" in c.lower() and "time" in c.lower() and "from" not in c.lower()), None)
        avc_col = next((c for c in df.columns if "avc" in c.lower() and "data" in c.lower()), None)
        fcast_col = next((c for c in df.columns if c.strip().lower() == "forecast"), None)
        actual_col = next((c for c in df.columns if "actual" in c.lower() and "generation" in c.lower()), None)

        if any(r is None for r in [date_col, block_col, from_col, to_col, avc_col, fcast_col, actual_col]):
            tqdm.write(f"WARNING: Agra missing required columns in {file.name}")
            continue

        base = df[[date_col, block_col, from_col, to_col, avc_col, fcast_col, actual_col]].copy()
        base = base.dropna(subset=[date_col])
        if base.empty:
            continue

        out = pd.DataFrame({
            "region": region,
            "plant_name": "TEQ Green-AGRA",
            "qca": qca,
            "date": pd.to_datetime(base[date_col], errors="coerce").dt.date,
            "time_block": pd.to_numeric(base[block_col], errors="coerce").fillna(1).astype(int).clip(1, 96),
            "from_time": base[from_col].astype(str).str.strip().str.slice(0, 5),
            "to_time": base[to_col].astype(str).str.strip().str.slice(0, 5),
            "avc": pd.to_numeric(base[avc_col], errors="coerce").fillna(0.0),
            "forecasted_power": pd.to_numeric(base[fcast_col], errors="coerce").fillna(0.0),
            "actual_power": pd.to_numeric(base[actual_col], errors="coerce").fillna(0.0),
            "ppa": float(ppa),
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
        date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
        block_col = next((c for c in df.columns if "block" in c.lower() and "no" in c.lower()), None)
        from_col = next((c for c in df.columns if "from" in c.lower() and "time" in c.lower()), None)
        to_col = next((c for c in df.columns if c.strip().lower() == "to time" or c.strip().startswith("TO TIME")), None)
        avc_col = next((c for c in df.columns if "avc" in c.lower() and "data" in c.lower()), None)
        fcast_col = next((c for c in df.columns if c.strip().lower() == "forecast"), None)
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
        base = df[cols].copy()
        base = base.dropna(subset=[date_col])
        if base.empty:
            continue

        avc_vals = pd.to_numeric(base[avc_col], errors="coerce").fillna(0.0) if avc_col else 0.0

        out = pd.DataFrame({
            "region": region,
            "plant_name": "TEQ Green-ASSAM",
            "qca": qca,
            "date": pd.to_datetime(base[date_col], errors="coerce").dt.date,
            "time_block": pd.to_numeric(base[block_col], errors="coerce").fillna(1).astype(int).clip(1, 96),
            "from_time": base[from_col].astype(str).str.strip().str.slice(0, 5),
            "to_time": base[to_col].astype(str).str.strip().str.slice(0, 5),
            "avc": avc_vals,
            "forecasted_power": pd.to_numeric(base[fcast_col], errors="coerce").fillna(0.0),
            "actual_power": pd.to_numeric(base[actual_col], errors="coerce").fillna(0.0),
            "ppa": float(ppa),
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

        ts = pd.to_datetime(raw["Time Stamp"], errors="coerce")
        raw = raw.loc[~ts.isna()].copy()
        if raw.empty:
            return pd.DataFrame()

        ts = pd.to_datetime(raw["Time Stamp"], errors="coerce")
        minutes = (ts.dt.hour.astype(int) * 60) + ts.dt.minute.astype(int)
        blocks = (minutes // 15) + 1

        forecast_col = "Accepted_Schedule_EOD [MW]" if "Accepted_Schedule_EOD [MW]" in raw.columns else None
        avc_col = "Reported AvC" if "Reported AvC" in raw.columns else None
        actual_col = "Actual [MW]" if "Actual [MW]" in raw.columns else None

        return pd.DataFrame({
            "region": region,
            "plant_name": "GHATODI",
            "qca": qca,
            "date": ts.dt.date,
            "time_block": blocks.astype(int),
            "from_time": ts.dt.strftime("%H:%M"),
            "to_time": (ts + pd.to_timedelta(15, unit="m")).dt.strftime("%H:%M"),
            "avc": raw[avc_col] if avc_col else 0.0,
            "forecasted_power": raw[forecast_col] if forecast_col else 0.0,
            "actual_power": raw[actual_col] if actual_col else 0.0,
            "ppa": float(ppa),
        })


def _read_50hz_table(con: duckdb.DuckDBPyConnection, *, path: Path, sheet: str) -> pd.DataFrame:
    """Read the data table from 50Hz sheet."""
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        tqdm.write(f"WARNING: 50Hz sheet missing columns for {plant_name}: {missing}")
        return pd.DataFrame()

    base = df[required].copy()
    base = base.dropna(subset=["Date & Time Block"])
    parsed = base["Date & Time Block"].apply(parse_time_to_master)
    out = pd.DataFrame(parsed.tolist(), columns=["date", "time_block", "from_time", "to_time"])
    out = out.dropna(subset=["date", "time_block"])

    out["region"] = region
    out["plant_name"] = plant_name
    out["qca"] = qca
    out["avc"] = base.loc[out.index, "AVC"].values
    out["forecasted_power"] = base.loc[out.index, "Forecasted Schedule (MAL)"].values
    out["actual_power"] = base.loc[out.index, actual_source_col].values
    out["ppa"] = float(ppa)
    return out.reset_index(drop=True)


def ingest_50hz(source_dir: Path, *, region: str) -> pd.DataFrame:
    """Ingest 50Hz STU data from Excel."""
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
            kud_sheet = next((s for s in sheet_names if "kudligi" in s.lower()), None) if sheet_names else "Kudligi(HYB)-O2(Solar)"

            if raj_sheet:
                try:
                    df = _read_50hz_table(con, path=file, sheet=raj_sheet)
                    frames.append(
                        _build_50hz_rows(
                            df,
                            region=region,
                            plant_name="PSEGPL-RAJPIMPRI",
                            qca="Manikaran",
                            ppa=5.0,
                            actual_source_col="Green Gen-Meter",
                        )
                    )
                except Exception as e:
                    tqdm.write(f"WARNING: 50Hz read failed: {file.name}::{raj_sheet} -> {e}")

            if kud_sheet:
                try:
                    df = _read_50hz_table(con, path=file, sheet=kud_sheet)
                    frames.append(
                        _build_50hz_rows(
                            df,
                            region=region,
                            plant_name="CEPPL SOLAR KUDLIGI",
                            qca="Manikaran",
                            ppa=3.5,
                            actual_source_col="Green Gen-SCADA",
                        )
                    )
                except Exception as e:
                    tqdm.write(f"WARNING: 50Hz read failed: {file.name}::{kud_sheet} -> {e}")

        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    finally:
        con.close()


def _clean_for_master(df: pd.DataFrame, *, region_default: str) -> pd.DataFrame:
    """Clean and normalize incoming STU data for master schema."""
    if df.empty:
        return df

    out = df.copy()
    if "region" not in out.columns:
        out["region"] = region_default
    out["region"] = out["region"].fillna(region_default).astype(str).str.strip().str.lower()
    out["plant_name"] = out.get("plant_name", "UNKNOWN").astype(str).str.strip()
    out["plant_name"] = out["plant_name"].replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})

    if "qca" not in out.columns:
        out["qca"] = None
    out["qca"] = out["qca"].astype(str).replace({"nan": None, "None": None})

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])
    out["time_block"] = pd.to_numeric(out["time_block"], errors="coerce").fillna(1).astype(int).clip(1, 96)

    if "from_time" not in out.columns:
        out["from_time"] = None
    if "to_time" not in out.columns:
        out["to_time"] = None
    missing_time = out["from_time"].isna() | out["to_time"].isna() | (out["from_time"].astype(str).str.len() < 4)
    if missing_time.any():
        ft_tt = out.loc[missing_time, "time_block"].apply(_block_to_time_range)
        out.loc[missing_time, "from_time"] = [x[0] for x in ft_tt]
        out.loc[missing_time, "to_time"] = [x[1] for x in ft_tt]

    out["from_time"] = out["from_time"].astype(str).str.slice(0, 5)
    out["to_time"] = out["to_time"].astype(str).str.slice(0, 5)

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
    """Merge cleaned STU dataframe into master DuckDB."""
    if df.empty:
        print("No STU data found.")
        return

    base = base_dir or _PROJECT_ROOT
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
            print("\n[STU] master.duckdb is in use. Stop the dashboard (Ctrl+C in its terminal) and run again.")
            raise
    try:
        pbar = tqdm(total=1, desc="Merging into master", unit="batch", leave=True)
        _ensure_master_table(conn, table)

        df = _clean_for_master(df, region_default=region_default)
        if df.empty:
            pbar.close()
            print("No usable STU rows after cleaning.")
            return

        cols = conn.execute(f"PRAGMA table_info({table})").fetchdf()
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
        existing = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {table} m
            WHERE EXISTS (
                SELECT 1 FROM incoming i
                WHERE m.region = i.region
                  AND m.plant_name = i.plant_name
                  AND m.date = i.date
                  AND m.time_block = i.time_block
            );
            """
        ).fetchone()[0]

        col_list = ", ".join(master_cols)
        conn.execute(
            f"""
            INSERT INTO {table} ({col_list})
            SELECT {col_list}
            FROM incoming i
            WHERE NOT EXISTS (
                SELECT 1 FROM {table} m
                WHERE m.region = i.region
                  AND m.plant_name = i.plant_name
                  AND m.date = i.date
                  AND m.time_block = i.time_block
            )
            ORDER BY i.region, i.plant_name, i.date, i.time_block;
            """
        )

        inserted = max(0, len(incoming) - int(existing))
        conn.unregister("incoming")
        pbar.update(1)
        pbar.close()

        print(f"Inserted {inserted:,} new rows into {resolved_db.name}::{table}")
        print(f"   Incoming rows (deduped): {len(incoming):,}")
        print(f"   Skipped (already exist): {min(len(incoming), int(existing)):,}")
    finally:
        conn.close()


def run_stu_ingestion(
    source_dir: str = "STU Raw",
    db_path: str = "master.duckdb",
    table: str = "master",
    region: str = "stu",
    motala_qca: str = "Reconnect",
    motala_ppa: float = 3.48,
    dry_run: bool = False,
    base_dir: Optional[Path] = None,
) -> None:
    """Run full STU ingestion pipeline."""
    base = base_dir or _PROJECT_ROOT
    source_path = _coerce_path(source_dir, base)
    db_path_resolved = _coerce_path(db_path, base)

    print("Loading STU data...")
    print(f"Source: {source_path}")
    print(f"Target: {db_path_resolved}::{table}")

    motala = ingest_motala(source_path, region=region, qca=motala_qca, ppa=motala_ppa)
    hz50 = ingest_50hz(source_path, region=region)
    agra = ingest_agra(source_path, region=region, qca="Reconnect", ppa=5.0)
    amguri = ingest_amguri(source_path, region=region, qca="Reconnect", ppa=3.798)
    ghatodi = ingest_ghatodi(source_path, region=region, qca="Reconnect", ppa=3.7)

    frames: List[pd.DataFrame] = []
    for d in (motala, hz50, agra, amguri, ghatodi):
        if not d.empty:
            frames.append(d)

    final_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print(f"Motala rows parsed: {len(motala):,}")
    print(f"50Hz rows parsed  : {len(hz50):,}")
    print(f"Agra rows parsed  : {len(agra):,}")
    print(f"Amguri rows parsed: {len(amguri):,}")
    print(f"Ghatodi rows parsed: {len(ghatodi):,}")
    print(f"Total rows parsed : {len(final_df):,}")

    if dry_run:
        cleaned = _clean_for_master(final_df, region_default=region)
        print(f"Dry-run: rows after cleaning/dedup: {len(cleaned):,}")
        return

    merge_into_master(
        final_df,
        db_path=db_path_resolved,
        table=table,
        region_default=region,
        base_dir=base,
    )

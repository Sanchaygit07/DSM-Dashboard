#!/usr/bin/env python3
"""
STU to Master DuckDB — CLI entry point for STU ingestion.

Usage:
    python stu_to_master.py [options]

Options:
    --source-dir   STU Raw source directory (default: STU Raw relative to project root,
                   or env DSM_STU_SOURCE)
    --master-db    Master DuckDB path (default: master.duckdb or env DSM_MASTER_DB_PATH)
    --table        Master table name (default: master)
    --region       Region tag for STU rows (default: stu)
    --dry-run      Parse and clean data but do NOT write to DB
    --help         Show this message

Environment overrides:
    DSM_STU_SOURCE      Path to STU Raw directory
    DSM_MASTER_DB_PATH  Path to master.duckdb

Examples:
    python stu_to_master.py
    python stu_to_master.py --dry-run
    python stu_to_master.py --source-dir "D:/Sanchay/DSM MASTER/STU Raw" --master-db "D:/Sanchay/DSM MASTER/master.duckdb"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Resolve project root so relative imports work whether run from any cwd ──
_SCRIPT_DIR  = Path(__file__).resolve().parent          # DSM MASTER/
_PROJECT_ROOT = _SCRIPT_DIR                             # same level
sys.path.insert(0, str(_PROJECT_ROOT))


def _resolve(env_var: str, arg_val: str | None, default_rel: str) -> str:
    """Priority: CLI arg > env var > default relative to project root."""
    if arg_val:
        return arg_val
    env = os.getenv(env_var)
    if env:
        return env
    return str(_PROJECT_ROOT / default_rel)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load STU raw Excel/CSV files into master.duckdb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="STU Raw source directory (default: DSM_STU_SOURCE env or 'STU Raw' in project root)",
    )
    parser.add_argument(
        "--master-db",
        default=None,
        help="Path to master.duckdb (default: DSM_MASTER_DB_PATH env or 'master.duckdb' in project root)",
    )
    parser.add_argument(
        "--table",
        default="master",
        help="Master table name (default: master)",
    )
    parser.add_argument(
        "--region",
        default="stu",
        help="Region tag written to the DB for all STU rows (default: stu)",
    )
    parser.add_argument(
        "--motala-qca",
        default="Reconnect",
        help="QCA value for Motala site (default: Reconnect)",
    )
    parser.add_argument(
        "--motala-ppa",
        type=float,
        default=3.48,
        help="PPA value for Motala site (default: 3.48)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + clean data but do NOT write to master.duckdb",
    )

    args = parser.parse_args()

    source_dir = _resolve("DSM_STU_SOURCE",     args.source_dir, "STU Raw")
    master_db  = _resolve("DSM_MASTER_DB_PATH", args.master_db,  "master.duckdb")

    # ── Try ingestion import from both flat and package layouts ──
    try:
        from ingestion.stu_ingestion import run_stu_ingestion  # package layout
    except ImportError:
        try:
            from stu_ingestion import run_stu_ingestion        # flat layout (same dir)
        except ImportError as e:
            print(f"[ERROR] Cannot import stu_ingestion: {e}")
            print("        Make sure stu_ingestion.py is in the same directory as this script,")
            print("        or inside an 'ingestion/' sub-package.")
            sys.exit(1)

    print("STU → Master DuckDB Ingestion")
    print(f"  Project root : {_PROJECT_ROOT}")
    print(f"  Source dir   : {source_dir}")
    print(f"  Master DB    : {master_db}")
    print(f"  Table        : {args.table}")
    print(f"  Region       : {args.region}")
    print(f"  Dry run      : {args.dry_run}")
    print()

    run_stu_ingestion(
        source_dir  = source_dir,
        db_path     = master_db,
        table       = args.table,
        region      = args.region,
        motala_qca  = args.motala_qca,
        motala_ppa  = args.motala_ppa,
        dry_run     = args.dry_run,
        base_dir    = _PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()  
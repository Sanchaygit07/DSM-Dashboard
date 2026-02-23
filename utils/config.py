"""
Configuration abstraction for DSM Analytics.

All paths can be overridden via environment variables for Docker/SaaS deployment.
See .env.example for the full list.

UI is stateless: dashboard reads from DuckDB on each request; dcc.Store holds client-side state only.

Environment variables (set in Render → Environment):
  DSM_MASTER_DB_PATH   Full path to master.duckdb
  DB_URL               Google Drive / HTTP URL to download the DB (used by download_db.py)
  DSM_NRPC_DB          Path to nrpc.duckdb
  DSM_SRPC_DB          Path to srpc.duckdb
  DSM_WRPC_DB          Path to wrpc.duckdb
  DSM_STU_SOURCE       Path to STU Raw source directory
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# config.py lives at: <project_root>/utils/config.py
# So parent      = utils/
#    parent.parent = <project_root>  (e.g. /app on Render, or DSM MASTER locally)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(env_var: str, default_name: str, *, must_exist: bool = False) -> str:
    """Resolve path: env override → project_root/default_name → cwd/default_name."""
    env = os.getenv(env_var)
    if env:
        return env
    candidate = _PROJECT_ROOT / default_name
    return str(candidate)


def get_project_root() -> Path:
    """Return project root directory."""
    return _PROJECT_ROOT


def get_master_db_path() -> str:
    """
    Return the absolute path to master.duckdb.

    Resolution order:
      1. Env var  DSM_MASTER_DB_PATH          (set this on Render)
      2. <project_root>/data/master.duckdb    (standard layout)
      3. <project_root>/master.duckdb         (legacy flat layout)
      4. data/master.duckdb                   (cwd-relative last resort)
    """
    # 1. Explicit env override (Render, Docker, CI)
    env = os.getenv("DSM_MASTER_DB_PATH")
    if env:
        return env

    # 2. Standard layout: data/ subfolder
    candidate_data = _PROJECT_ROOT / "data" / "master.duckdb"
    if candidate_data.exists():
        return str(candidate_data)

    # 3. Legacy flat layout: DB at project root
    candidate_root = _PROJECT_ROOT / "master.duckdb"
    if candidate_root.exists():
        return str(candidate_root)

    # 4. Last resort: return the expected path even if it doesn't exist yet
    #    (download_db.py will create it here before the app needs it)
    return str(_PROJECT_ROOT / "data" / "master.duckdb")


def get_nrpc_db_path() -> str:
    """Path to NRPC DuckDB. Override with DSM_NRPC_DB."""
    return _resolve_path("DSM_NRPC_DB", "nrpc.duckdb")


def get_srpc_db_path() -> str:
    """Path to SRPC DuckDB. Override with DSM_SRPC_DB."""
    return _resolve_path("DSM_SRPC_DB", "srpc.duckdb")


def get_wrpc_db_path() -> str:
    """Path to WRPC DuckDB. Override with DSM_WRPC_DB."""
    return _resolve_path("DSM_WRPC_DB", "wrpc.duckdb")


def get_stu_source_dir() -> str:
    """Path to STU Raw source directory. Override with DSM_STU_SOURCE."""
    return _resolve_path("DSM_STU_SOURCE", "STU Raw")
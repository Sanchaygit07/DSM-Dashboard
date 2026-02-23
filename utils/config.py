"""
Configuration abstraction for DSM Analytics.

All paths can be overridden via environment variables for Docker/SaaS deployment.
See .env.example for the full list.

UI is stateless: dashboard reads from DuckDB on each request; dcc.Store holds client-side state only.
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(env_var: str, default_name: str, *, must_exist: bool = False) -> str:
    """Resolve path: env override, else project_root / default_name."""
    env = os.getenv(env_var)
    if env:
        return env
    candidate = _PROJECT_ROOT / default_name
    if must_exist and not candidate.exists():
        return str(_PROJECT_ROOT / default_name)  # Return anyway for creation
    return str(candidate)


def get_project_root() -> Path:
    """Return project root (DSM MASTER folder)."""
    return _PROJECT_ROOT


def get_master_db_path() -> str:
    """
    Return path to master DuckDB file.

    Priority:
    - env `DSM_MASTER_DB_PATH`
    - `data/master.duckdb` in project root
    - `master.duckdb` in project root (legacy)
    - fallback `data/master.duckdb` (cwd-relative)
    """
    env = os.getenv("DSM_MASTER_DB_PATH")
    if env:
        return env
    candidate_data = _PROJECT_ROOT / "data" / "master.duckdb"
    if candidate_data.exists():
        return str(candidate_data)
    candidate_root = _PROJECT_ROOT / "master.duckdb"
    if candidate_root.exists():
        return str(candidate_root)
    return "data/master.duckdb"


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


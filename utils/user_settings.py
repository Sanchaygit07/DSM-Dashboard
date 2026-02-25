"""
user_settings.py — Persistent user-settings storage backed by master.duckdb.

A new table  `user_settings`  is created inside the existing master.duckdb:

    CREATE TABLE user_settings (
        setting_key   VARCHAR PRIMARY KEY,
        setting_value VARCHAR NOT NULL,
        updated_at    TIMESTAMP NOT NULL
    );

Keys used by the dashboard
──────────────────────────
  "default_settings"   → JSON blob  {err_mode, x_pct, bands, zero_basis_guard, ...}
  "preset:<name>"      → JSON blob  {name, err_mode, x_pct, bands, ...}

Public API (used in dsm_dashboard.py callbacks)
───────────────────────────────────────────────
  load_default_settings(db_path)  → dict | None
  save_default_settings(db_path, settings: dict) -> bool
  list_presets(db_path)           -> list[dict]   # [{name, settings}, ...]
  save_preset(db_path, name, settings: dict) -> bool
  delete_preset(db_path, name)    -> bool
  load_preset(db_path, name)      -> dict | None
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb


_DEFAULT_KEY   = "default_settings"
_PRESET_PREFIX = "preset:"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_rw(db_path: str) -> duckdb.DuckDBPyConnection:
    """Open master.duckdb in read-write mode with simple retry."""
    for attempt in range(3):
        try:
            return duckdb.connect(db_path, read_only=False)
        except Exception as e:
            err = str(e).lower()
            if ("cannot open" in err or "in use" in err or "another process" in err) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def _ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            setting_key   VARCHAR PRIMARY KEY,
            setting_value VARCHAR NOT NULL,
            updated_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
        );
    """)


def _upsert(conn: duckdb.DuckDBPyConnection, key: str, value_json: str) -> None:
    """INSERT or UPDATE a single row."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    # DuckDB supports INSERT OR REPLACE
    conn.execute("""
        INSERT INTO user_settings (setting_key, setting_value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT (setting_key) DO UPDATE
          SET setting_value = excluded.setting_value,
              updated_at    = excluded.updated_at;
    """, [key, value_json, now])


def _fetch(conn: duckdb.DuckDBPyConnection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT setting_value FROM user_settings WHERE setting_key = ?", [key]
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_default_settings(db_path: str) -> Optional[Dict[str, Any]]:
    """
    Load the saved default settings dict from the DB.
    Returns None if nothing has been saved yet.
    """
    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            # Table may not exist yet on first run
            tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
            if "user_settings" not in tables:
                return None
            raw = _fetch(conn, _DEFAULT_KEY)
        return json.loads(raw) if raw else None
    except Exception as e:
        print(f"[user_settings] load_default_settings error: {e}")
        return None


def save_default_settings(db_path: str, settings: Dict[str, Any]) -> bool:
    """
    Persist the default settings dict to the DB.
    Returns True on success.
    """
    try:
        conn = _open_rw(db_path)
        try:
            _ensure_table(conn)
            _upsert(conn, _DEFAULT_KEY, json.dumps(settings, default=str))
        finally:
            conn.close()
        return True
    except Exception as e:
        print(f"[user_settings] save_default_settings error: {e}")
        return False


def list_presets(db_path: str) -> List[Dict[str, Any]]:
    """
    Return all saved presets as a list of {name, settings, updated_at}.
    """
    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
            if "user_settings" not in tables:
                return []
            rows = conn.execute(
                "SELECT setting_key, setting_value, updated_at "
                "FROM user_settings "
                f"WHERE setting_key LIKE '{_PRESET_PREFIX}%' "
                "ORDER BY updated_at DESC"
            ).fetchall()
        result = []
        for key, val, ts in rows:
            name = key[len(_PRESET_PREFIX):]
            try:
                settings = json.loads(val)
            except Exception:
                settings = {}
            result.append({"name": name, "settings": settings, "updated_at": str(ts)})
        return result
    except Exception as e:
        print(f"[user_settings] list_presets error: {e}")
        return []


def save_preset(db_path: str, name: str, settings: Dict[str, Any]) -> bool:
    """
    Save (create or overwrite) a named preset.
    The name is stored as the key suffix after 'preset:'.
    Returns True on success.
    """
    if not name or not name.strip():
        print("[user_settings] save_preset: preset name cannot be empty.")
        return False
    key = _PRESET_PREFIX + name.strip()
    payload = {**settings, "name": name.strip()}
    try:
        conn = _open_rw(db_path)
        try:
            _ensure_table(conn)
            _upsert(conn, key, json.dumps(payload, default=str))
        finally:
            conn.close()
        return True
    except Exception as e:
        print(f"[user_settings] save_preset error: {e}")
        return False


def delete_preset(db_path: str, name: str) -> bool:
    """
    Delete a named preset from the DB.
    Returns True if the row was found and deleted.
    """
    key = _PRESET_PREFIX + (name or "").strip()
    try:
        conn = _open_rw(db_path)
        try:
            _ensure_table(conn)
            conn.execute("DELETE FROM user_settings WHERE setting_key = ?", [key])
        finally:
            conn.close()
        return True
    except Exception as e:
        print(f"[user_settings] delete_preset error: {e}")
        return False


def load_preset(db_path: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Load a single named preset dict.
    Returns None if not found.
    """
    key = _PRESET_PREFIX + (name or "").strip()
    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
            if "user_settings" not in tables:
                return None
            raw = _fetch(conn, key)
        return json.loads(raw) if raw else None
    except Exception as e:
        print(f"[user_settings] load_preset error: {e}")
        return None
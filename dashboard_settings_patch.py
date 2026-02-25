"""
dashboard_settings_patch.py
════════════════════════════
This file shows the EXACT changes to make inside dsm_dashboard.py so that
"Save Settings" and "Save as Preset" write to master.duckdb (via user_settings.py)
instead of only to dcc.Store (which is lost on refresh / new machine).

The current dcc.Store behaviour is KEPT as a fast in-session cache —
the DB is the source of truth that is loaded on startup.

HOW TO APPLY
────────────
1. Copy user_settings.py into D:\Sanchay\DSM MASTER\utils\user_settings.py
   (or the same folder as dsm_dashboard.py — both work).
2. Apply the two import lines and the four callback patches described below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH 1 — Add import near the top of dsm_dashboard.py
         (after the existing 'from utils.config import ...' line)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── ADD THESE LINES right after: from utils.config import get_master_db_path ──
# ──────────────────────────────────────────────────────────────────────────────
_PATCH_1_IMPORT = """
try:
    from utils.user_settings import (
        load_default_settings,
        save_default_settings,
        list_presets,
        save_preset,
        delete_preset,
        load_preset,
    )
    _SETTINGS_BACKEND = "duckdb"
except ImportError:
    # Graceful fallback: settings only survive in the browser session
    _SETTINGS_BACKEND = "memory"
    def load_default_settings(db_path): return None
    def save_default_settings(db_path, s): return False
    def list_presets(db_path): return []
    def save_preset(db_path, n, s): return False
    def delete_preset(db_path, n): return False
    def load_preset(db_path, n): return None
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH 2 — Replace the save_settings callback (search for def save_settings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_PATCH_2_OLD = '''
def save_settings(n_clicks, err_mode, x_pct, bands_data, zero_basis_guard):
    """Save current settings to local storage"""
    if not n_clicks:
        raise PreventUpdate
    
    # Package settings
    settings = {
        "err_mode": err_mode,
        "x_pct": x_pct,
        "bands": bands_data,
        "zero_basis_guard": "on" in (zero_basis_guard or [])
    }
    
    # Success message
    message = dbc.Alert([
        html.Span("✓ ", style={"fontSize": "1.2rem", "marginRight": "8px"}),
        html.Strong("Settings saved successfully!"),
        html.Br(),
        html.Small("Your configuration will be used when running analysis.")
    ], color="success", dismissable=True, duration=4000)
    
    return message, settings
'''

_PATCH_2_NEW = '''
def save_settings(n_clicks, err_mode, x_pct, bands_data, zero_basis_guard):
    """Save current settings — persisted to master.duckdb AND browser dcc.Store."""
    if not n_clicks:
        raise PreventUpdate

    settings = {
        "err_mode": err_mode,
        "x_pct": x_pct,
        "bands": bands_data,
        "zero_basis_guard": "on" in (zero_basis_guard or []),
    }

    # ── Persist to DuckDB (survives refresh / other machines) ──
    db_saved = save_default_settings(get_master_db_path(), settings)
    extra    = " (saved to DB ✓)" if db_saved else " (DB unavailable — session only)"

    message = dbc.Alert([
        html.Span("✓ ", style={"fontSize": "1.2rem", "marginRight": "8px"}),
        html.Strong("Settings saved successfully!" + extra),
        html.Br(),
        html.Small("Your configuration will be used when running analysis.")
    ], color="success", dismissable=True, duration=4000)

    return message, settings
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH 3 — Replace the startup initialiser for saved-settings-store
#
# Find the callback that loads saved-settings on startup.
# It currently returns defaults.  Replace with DB-first lookup.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ADD a new callback right after the existing save_settings callback:
_PATCH_3_NEW_CALLBACK = '''
# ─── Load saved-settings from DuckDB on app startup ───────────────────────
@app.callback(
    Output("saved-settings-store", "data", allow_duplicate=True),
    Output("presets-store",         "data", allow_duplicate=True),
    Input("nav-store", "data"),          # fires on every navigation
    prevent_initial_call=False,
)
def hydrate_settings_from_db(nav_data):
    """
    On every page load / navigation, pull the latest settings and presets
    from master.duckdb into the browser dcc.Store so all callbacks get them.
    This makes settings work across machines sharing the same DB file.
    """
    db_path = get_master_db_path()

    # Default settings
    default_settings = load_default_settings(db_path)
    if not isinstance(default_settings, dict):
        default_settings = {}   # first run — stay with empty → DEFAULT_BANDS used downstream

    # Presets
    raw_presets = list_presets(db_path)            # [{name, settings, updated_at}, ...]
    presets_for_store = [
        {"name": p["name"], "settings": p["settings"]}
        for p in raw_presets
    ]

    return default_settings, presets_for_store
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATCH 4 — Replace the "Save as Preset" / "Delete Preset" callback
#
# Search for the callback that has:
#   Input("btn-save-preset",   "n_clicks"),
#   Input("btn-delete-preset", "n_clicks"),
# and patch its body to also call save_preset() / delete_preset().
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# The simplest way: find "btn-save-preset" callback body and ADD the DB call.
# Locate the line that does:
#     presets_store = (presets_store or []) + [new_preset]
# and add right after it:
_PATCH_4_SAVE_PRESET_ADDITION = """
    # ── Also persist to DuckDB ──
    save_preset(get_master_db_path(), preset_name, new_preset.get("settings", {}))
"""

# Locate the delete block that removes a preset from the list:
#     updated = [p for p in presets_store if p.get("name") not in names_to_delete]
# and add right after it:
_PATCH_4_DELETE_PRESET_ADDITION = """
    # ── Also delete from DuckDB ──
    for n in names_to_delete:
        delete_preset(get_master_db_path(), n)
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK VERIFICATION
# Run this snippet in a Python shell to confirm the table was created:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_VERIFY_SNIPPET = """
import duckdb
con = duckdb.connect(r"D:\\Sanchay\\DSM MASTER\\master.duckdb", read_only=True)
print(con.execute("SELECT * FROM user_settings").fetchdf())
con.close()
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nPATCH 1 — Import block:\n", _PATCH_1_IMPORT)
    print("\nPATCH 2 — save_settings OLD:\n", _PATCH_2_OLD)
    print("\nPATCH 2 — save_settings NEW:\n", _PATCH_2_NEW)
    print("\nPATCH 3 — hydrate_settings_from_db (new callback):\n", _PATCH_3_NEW_CALLBACK)
    print("\nPATCH 4 — Save preset DB call addition:\n", _PATCH_4_SAVE_PRESET_ADDITION)
    print("\nPATCH 4 — Delete preset DB call addition:\n", _PATCH_4_DELETE_PRESET_ADDITION)
    print("\nVerify with:\n", _VERIFY_SNIPPET)
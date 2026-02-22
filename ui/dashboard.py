"""
Dashboard module — re-exports the main DSM Analytics app.

The main app lives in dsm_dashboard.py at project root. This module provides
a unified import path for the UI layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when importing as ui.dashboard
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dsm_dashboard import app, server  # noqa: E402

__all__ = ["app", "server"]

"""
download_db.py
==============
Downloads (or locates) the master DuckDB file before the app starts.

Priority order:
  1. DB already exists at DB_PATH → skip download
  2. DB_URL env var is set        → download from that URL (Google Drive or direct)
  3. Neither                      → print a clear warning and continue
     (the app will show empty dropdowns but won't crash on startup)

Environment variables (set these in Render → Environment):
  DSM_MASTER_DB_PATH   Path where the DB should be stored.
                       Default: <project_root>/data/master.duckdb
  DB_URL               Public download URL for the DB file.
                       Supports:
                         • Google Drive share link:
                             https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
                         • Google Drive direct export link:
                             https://drive.google.com/uc?export=download&id=<FILE_ID>
                         • Any direct HTTP/HTTPS URL to a .duckdb file
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent

DB_PATH = Path(
    os.getenv("DSM_MASTER_DB_PATH", str(_PROJECT_ROOT / "data" / "master.duckdb"))
)

# Raw URL from environment (Google Drive share link OR direct download URL)
_RAW_URL: str | None = os.getenv("DB_URL")


# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------
_GDRIVE_SHARE_RE = re.compile(
    r"https://drive\.google\.com/file/d/([^/]+)/", re.IGNORECASE
)
_GDRIVE_UC_RE = re.compile(
    r"https://drive\.google\.com/uc\?.*id=([^&]+)", re.IGNORECASE
)


def _extract_gdrive_id(url: str) -> str | None:
    """Extract Google Drive file ID from share or export link."""
    for pattern in (_GDRIVE_SHARE_RE, _GDRIVE_UC_RE):
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


def _gdrive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"


def _get_confirmation_token(response: requests.Response) -> str | None:
    """
    Google Drive shows an HTML warning page for large files.
    This extracts the confirm token so we can retry with it.
    """
    # Check for NID cookie (older GDrive behaviour)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    # Newer GDrive: look for confirm=<TOKEN> in the redirect HTML
    match = re.search(r'confirm=([0-9A-Za-z_\-]+)', response.text)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Core download
# ---------------------------------------------------------------------------
def _download_from_url(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest*, handling Google Drive quirks."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # Detect Google Drive and convert to direct-download URL
    file_id = _extract_gdrive_id(url)
    if file_id:
        print(f"[download_db] Detected Google Drive file ID: {file_id}")
        download_url = _gdrive_direct_url(file_id)
    else:
        download_url = url

    print(f"[download_db] Requesting: {download_url}")
    response = session.get(download_url, stream=True, timeout=120)
    response.raise_for_status()

    # Handle Google Drive large-file confirmation page
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type and file_id:
        print("[download_db] Got HTML response from Google Drive — handling confirmation page...")
        token = _get_confirmation_token(response)
        if token:
            confirmed_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
            print(f"[download_db] Retrying with confirm token: {confirmed_url}")
            response = session.get(confirmed_url, stream=True, timeout=300)
            response.raise_for_status()
        else:
            # Try the export/download endpoint as last resort
            alt_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            print(f"[download_db] No token found; trying alt URL: {alt_url}")
            response = session.get(alt_url, stream=True, timeout=300)
            response.raise_for_status()

    # Verify we got binary data, not another HTML page
    final_type = response.headers.get("Content-Type", "")
    if "text/html" in final_type:
        snippet = response.text[:500]
        raise RuntimeError(
            f"[download_db] Still receiving HTML instead of binary after confirmation. "
            f"Check that your Google Drive file is publicly shared (Anyone with the link → Viewer).\n"
            f"Response snippet: {snippet}"
        )

    # Stream to disk
    total = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    print(f"[download_db] Download complete — {total / 1_048_576:.1f} MB written to {dest}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def download_db() -> None:
    """
    Ensure master.duckdb exists at DB_PATH.
    Called automatically when dsm_dashboard.py starts.
    """
    # 1. Already present — nothing to do
    if DB_PATH.exists():
        size_mb = DB_PATH.stat().st_size / 1_048_576
        print(f"[download_db] DB already present at {DB_PATH} ({size_mb:.1f} MB) — skipping download.")
        return

    # 2. Try downloading from DB_URL
    if _RAW_URL:
        print(f"[download_db] DB not found at {DB_PATH}. Downloading from DB_URL...")
        try:
            _download_from_url(_RAW_URL, DB_PATH)
            return
        except Exception as exc:
            print(f"[download_db] ERROR during download: {exc}", file=sys.stderr)
            raise

    # 3. Neither DB nor URL — warn clearly
    print(
        "[download_db] WARNING: master.duckdb not found and DB_URL is not set.\n"
        "  The app will start but Region/Plant dropdowns will be empty.\n"
        "  Fix options:\n"
        "    a) Set DB_URL env var in Render to your Google Drive share link, OR\n"
        "    b) Commit data/master.duckdb to your repository (if file size allows).",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Allow running standalone: python download_db.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[download_db] DB target path : {DB_PATH}")
    print(f"[download_db] DB_URL         : {_RAW_URL or '(not set)'}")
    download_db()
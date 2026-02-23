"""
download_db.py
==============
Downloads (or locates) the master DuckDB file before the app starts.

Priority order:
  1. DB already exists at DB_PATH  →  skip download
  2. DB_URL env var is set          →  download via gdown (Google Drive) or requests (direct URL)
  3. Neither                        →  print a clear warning and continue
     (the app will start but dropdowns will be empty)

Environment variables (set in Render → Environment):
  DSM_MASTER_DB_PATH   Path where the DB should be stored.
                       Default: <project_root>/data/master.duckdb
  DB_URL               Public download URL for the DB file.
                       Supported formats:
                         • Google Drive share link:
                             https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
                         • Google Drive open link:
                             https://drive.google.com/open?id=<FILE_ID>
                         • Any direct HTTP/HTTPS URL to a .duckdb binary

IMPORTANT: Make sure gdown>=5.1.0 is listed in requirements.txt
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

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
_GDRIVE_PATTERNS = [
    re.compile(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", re.IGNORECASE),
    re.compile(r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)", re.IGNORECASE),
    re.compile(r"drive\.google\.com/uc\?.*id=([a-zA-Z0-9_-]+)", re.IGNORECASE),
]


def _extract_gdrive_id(url: str) -> str | None:
    """Extract Google Drive file ID from any supported GDrive URL format."""
    for pattern in _GDRIVE_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Download backends
# ---------------------------------------------------------------------------

def _download_with_gdown(file_id: str, dest: Path) -> None:
    """
    Download a Google Drive file using gdown.
    gdown handles virus scan warnings and large file confirmations automatically.
    Requires: gdown>=5.1.0 in requirements.txt
    """
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "[download_db] gdown is not installed.\n"
            "  Add 'gdown>=5.1.0' to your requirements.txt and redeploy."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    gdrive_url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download_db] Google Drive file ID  : {file_id}")
    print(f"[download_db] Downloading to        : {dest}")

    output = gdown.download(
        gdrive_url,
        str(dest),
        quiet=False,
        fuzzy=True,          # handles various GDrive URL formats
    )

    if output is None or not dest.exists():
        raise RuntimeError(
            f"[download_db] gdown returned None — download failed.\n"
            f"  Possible causes:\n"
            f"    1. File is not shared as 'Anyone with the link → Viewer'\n"
            f"    2. File ID is incorrect (check your DB_URL env var)\n"
            f"    3. File has been moved or deleted in Google Drive\n"
            f"  File ID used: {file_id}"
        )

    size_mb = dest.stat().st_size / 1_048_576
    if size_mb < 0.001:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"[download_db] Downloaded file is empty (0 bytes).\n"
            f"  The Google Drive link may be pointing to an HTML page instead of the binary.\n"
            f"  Verify the file is publicly shared and the File ID is correct."
        )

    print(f"[download_db] Download complete — {size_mb:.1f} MB written to {dest}")


def _download_direct(url: str, dest: Path) -> None:
    """
    Download from a plain HTTP/HTTPS URL (non-Google-Drive: Dropbox, S3, etc.).
    Falls back to this when DB_URL is not a Google Drive link.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "[download_db] requests is not installed.\n"
            "  Add 'requests' to requirements.txt."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_db] Downloading from: {url}")
    print(f"[download_db] Destination     : {dest}")

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(
                f"[download_db] Received HTML instead of binary.\n"
                f"  URL: {url}\n"
                f"  Content-Type: {content_type}\n"
                f"  Make sure the URL is a direct download link, not a web page."
            )
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
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
    Called automatically at app startup from dsm_dashboard.py.
    """
    # 1. Already present — nothing to do
    if DB_PATH.exists():
        size_mb = DB_PATH.stat().st_size / 1_048_576
        print(f"[download_db] DB already present at {DB_PATH} ({size_mb:.1f} MB) — skipping download.")
        return

    # 2. Try downloading from DB_URL
    if _RAW_URL:
        print(f"[download_db] DB not found at {DB_PATH}.")
        file_id = _extract_gdrive_id(_RAW_URL)
        if file_id:
            # Google Drive URL → use gdown
            _download_with_gdown(file_id, DB_PATH)
        else:
            # Plain HTTP URL (Dropbox, S3, Azure Blob, etc.) → use requests
            _download_direct(_RAW_URL, DB_PATH)
        return

    # 3. Neither DB nor URL found — warn clearly but don't crash
    print(
        "[download_db] WARNING: master.duckdb not found and DB_URL is not set.\n"
        "  The app will start but Region/Plant dropdowns will be empty.\n"
        "  To fix, set DB_URL in Render → Environment to your Google Drive share link:\n"
        "    https://drive.google.com/file/d/<YOUR_FILE_ID>/view?usp=sharing",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Allow running standalone for testing: python download_db.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[download_db] DB target path : {DB_PATH}")
    print(f"[download_db] DB_URL         : {_RAW_URL or '(not set)'}")
    download_db()
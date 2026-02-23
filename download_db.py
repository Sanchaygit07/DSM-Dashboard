import os
from pathlib import Path

import requests

# Local Windows path where your DB currently exists (local development convenience)
LOCAL_DB_PATH = Path(r"C:\Users\Lenovo\Desktop\Project\sanchay_development\DSM MASTER\master.duckdb")

_PROJECT_ROOT = Path(__file__).resolve().parent

# Path where the app expects the DB inside the project (overrideable for deployment)
DB_PATH = Path(os.getenv("DSM_MASTER_DB_PATH", str(_PROJECT_ROOT / "data" / "master.duckdb")))

# Optional Google Drive fallback (for Render deployment)
GDRIVE_URL = os.getenv("DB_URL")


def download_db():
    # If DB already exists in project folder, skip
    if DB_PATH.exists():
        print("DB already exists in project folder, skipping.")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # First try copying from local Windows path (for local development)
    if LOCAL_DB_PATH.exists():
        print("Copying DB from local Windows directory...")
        with open(LOCAL_DB_PATH, "rb") as src, open(DB_PATH, "wb") as dst:
            dst.write(src.read())
        print("Local DB copy completed!")
        return

    # Fallback to Google Drive (for Render deployment)
    if not GDRIVE_URL:
        raise ValueError("DB not found locally and DB_URL not set!")

    print("Downloading DuckDB from Google Drive...")
    with requests.get(GDRIVE_URL, stream=True) as r:
        r.raise_for_status()
        with open(DB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Download completed!")


if __name__ == "__main__":
    download_db()
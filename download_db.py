import os
import requests

# Local Windows path where your DB currently exists
LOCAL_DB_PATH = r"C:\Users\Lenovo\Desktop\Project\sanchay_development\DSM MASTER\master.duckdb"

# Path where the app expects the DB inside the project
DB_PATH = "data/master.duckdb"

# Optional Google Drive fallback
GDRIVE_URL = os.getenv("DB_URL")


def download_db():
    # If DB already exists in project folder, skip
    if os.path.exists(DB_PATH):
        print("DB already exists in project folder, skipping.")
        return

    os.makedirs("data", exist_ok=True)

    # First try copying from local Windows path (for local development)
    if os.path.exists(LOCAL_DB_PATH):
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
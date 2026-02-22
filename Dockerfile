# DSM Analytics – Production Ready Dockerfile

FROM python:3.11-slim

# Working directory
WORKDIR /app

# Install system deps (Excel + DuckDB safety)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Ensure data directory exists
RUN mkdir -p data

# Environment config
ENV DSM_HOST=0.0.0.0
ENV DSM_PORT=8050
ENV DSM_MASTER_DB_PATH=data/master.duckdb

# Expose dashboard port
EXPOSE 8050

CMD python download_db.py && python dsm_dashboard.py
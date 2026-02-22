"""
RPC ingestion: merge NRPC, SRPC, WRPC DuckDB databases into a unified Master database.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_master_table(conn: duckdb.DuckDBPyConnection, table: str = "master") -> None:
    """Ensure the master table exists with proper schema and index."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            region VARCHAR NOT NULL,
            plant_name VARCHAR NOT NULL,
            date DATE NOT NULL,
            time_block INTEGER NOT NULL,
            from_time VARCHAR NOT NULL,
            to_time VARCHAR NOT NULL,
            avc DOUBLE NOT NULL,
            forecasted_power DOUBLE NOT NULL,
            actual_power DOUBLE NOT NULL,
            ppa DOUBLE NOT NULL
        )
    """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table}_key ON {table} (region, plant_name, date, time_block)"
    )


def _get_database_stats(conn: duckdb.DuckDBPyConnection, db_name: str) -> dict:
    """Get statistics from a regional database."""
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_name = None
        for table in tables:
            if table[0].lower() in ["nrpc", "srpc", "wrpc"]:
                table_name = table[0]
                break
        if not table_name:
            return {"error": f"No regional table found in {db_name}"}

        result = conn.execute(
            f"""
            SELECT
                COUNT(*) as total_rows,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT plant_name) as unique_plants,
                COUNT(DISTINCT region) as unique_regions
            FROM {table_name}
        """
        ).fetchone()

        return {
            "table": table_name,
            "total_rows": result[0],
            "min_date": result[1],
            "max_date": result[2],
            "unique_plants": result[3],
            "unique_regions": result[4],
        }
    except Exception as e:
        return {"error": str(e)}


def _merge_regional_data(
    regional_conn: duckdb.DuckDBPyConnection,
    master_conn: duckdb.DuckDBPyConnection,
    region_name: str,
    master_table: str = "master",
) -> dict:
    """Merge data from a regional database into the master database."""
    try:
        tables = regional_conn.execute("SHOW TABLES").fetchall()
        regional_table = None
        for table in tables:
            if table[0].lower() == region_name.lower():
                regional_table = table[0]
                break
        if not regional_table:
            return {"error": f"Table {region_name} not found in regional database"}

        total_count = regional_conn.execute(f"SELECT COUNT(*) FROM {regional_table}").fetchone()[0]
        if total_count == 0:
            return {"total_rows": 0, "inserted_rows": 0, "skipped_rows": 0}

        batch_size = 50000
        total_inserted = 0
        total_skipped = 0

        with tqdm(total=total_count, desc=f"Processing {region_name.upper()} data", unit="rows") as pbar:
            offset = 0
            while offset < total_count:
                batch_df = regional_conn.execute(
                    f"""
                    SELECT * FROM {regional_table}
                    ORDER BY region, plant_name, date, time_block
                    LIMIT {batch_size} OFFSET {offset}
                """
                ).df()

                if batch_df.empty:
                    break

                master_conn.register("incoming_batch", batch_df)

                existing_count = master_conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {master_table} m
                    WHERE EXISTS (
                        SELECT 1 FROM incoming_batch i
                        WHERE m.region = i.region
                        AND m.plant_name = i.plant_name
                        AND m.date = i.date
                        AND m.time_block = i.time_block
                    )
                """
                ).fetchone()[0]

                if existing_count < len(batch_df):
                    new_count = master_conn.execute(
                        f"""
                        SELECT COUNT(*) FROM incoming_batch i
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {master_table} m
                            WHERE m.region = i.region
                            AND m.plant_name = i.plant_name
                            AND m.date = i.date
                            AND m.time_block = i.time_block
                        )
                    """
                    ).fetchone()[0]
                    if new_count > 0:
                        master_conn.execute(
                            f"""
                            INSERT INTO {master_table}
                            SELECT * FROM incoming_batch i
                            WHERE NOT EXISTS (
                                SELECT 1 FROM {master_table} m
                                WHERE m.region = i.region
                                AND m.plant_name = i.plant_name
                                AND m.date = i.date
                                AND m.time_block = i.time_block
                            )
                            ORDER BY i.region, i.plant_name, i.date, i.time_block
                        """
                        )
                        total_inserted += new_count
                    total_skipped += len(batch_df) - new_count
                else:
                    total_skipped += len(batch_df)

                master_conn.unregister("incoming_batch")
                pbar.update(len(batch_df))
                pbar.set_postfix(inserted=f"{total_inserted:,}", skipped=f"{total_skipped:,}")
                offset += batch_size

        return {
            "total_rows": total_count,
            "inserted_rows": total_inserted,
            "skipped_rows": total_skipped,
        }
    except Exception as e:
        return {"error": str(e)}


def build_master_duckdb(
    nrpc_db_path: str,
    srpc_db_path: str,
    wrpc_db_path: str,
    master_db_path: str,
    master_table: str = "master",
    base_dir: Optional[Path] = None,
) -> None:
    """Build Master DuckDB by merging data from all regional databases."""
    base = base_dir or _PROJECT_ROOT

    def resolve(p: str) -> str:
        return str(base / p) if not os.path.isabs(p) else p

    nrpc_db_path = resolve(nrpc_db_path)
    srpc_db_path = resolve(srpc_db_path)
    wrpc_db_path = resolve(wrpc_db_path)
    master_db_path = resolve(master_db_path)

    print(f"Building Master DuckDB: {master_db_path}")
    print(f"NRPC Database: {nrpc_db_path}")
    print(f"SRPC Database: {srpc_db_path}")
    print(f"WRPC Database: {wrpc_db_path}")
    print(f"Master Table: {master_table}")
    print()

    regional_dbs = [
        (nrpc_db_path, "NRPC"),
        (srpc_db_path, "SRPC"),
        (wrpc_db_path, "WRPC"),
    ]

    for db_path, region_name in regional_dbs:
        if not os.path.exists(db_path):
            print(f"Error: {region_name} database does not exist: {db_path}")
            sys.exit(1)

    for attempt in range(3):
        try:
            master_conn = duckdb.connect(master_db_path)
            break
        except Exception as e:
            err = str(e).lower()
            if ("cannot open" in err or "in use" in err or "another process" in err) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print("\n[RPC] master.duckdb is in use. Stop the dashboard (Ctrl+C in its terminal) and run again.")
            raise

    try:
        _ensure_master_table(master_conn, master_table)

        print("=== Regional Database Statistics ===")
        regional_stats = {}
        for db_path, region_name in regional_dbs:
            try:
                regional_conn = duckdb.connect(db_path)
                stats = _get_database_stats(regional_conn, region_name)
                regional_stats[region_name.lower()] = stats

                if "error" in stats:
                    print(f"{region_name}: ERROR - {stats['error']}")
                else:
                    print(f"{region_name}:")
                    print(f"  Table: {stats['table']}")
                    print(f"  Total Rows: {stats['total_rows']:,}")
                    print(f"  Date Range: {stats['min_date']} to {stats['max_date']}")
                    print(f"  Unique Plants: {stats['unique_plants']}")
                    print(f"  Unique Regions: {stats['unique_regions']}")
                regional_conn.close()
            except Exception as e:
                print(f"{region_name}: ERROR - {str(e)}")
                regional_stats[region_name.lower()] = {"error": str(e)}

        print()
        print("=== Merging Regional Data ===")
        merge_results = {}

        for db_path, region_name in regional_dbs:
            region_key = region_name.lower()
            if "error" in regional_stats[region_key]:
                print(f"Skipping {region_name} due to errors")
                continue

            print(f"\nMerging {region_name} data...")
            try:
                regional_conn = duckdb.connect(db_path)
                result = _merge_regional_data(
                    regional_conn, master_conn, region_key, master_table
                )
                regional_conn.close()
                merge_results[region_key] = result

                if "error" in result:
                    print(f"Error merging {region_name}: {result['error']}")
                else:
                    print(f"{region_name} merge complete:")
                    print(f"  Total rows processed: {result['total_rows']:,}")
                    print(f"  New rows inserted: {result['inserted_rows']:,}")
                    print(f"  Rows skipped (already exist): {result['skipped_rows']:,}")
            except Exception as e:
                print(f"Error processing {region_name}: {str(e)}")
                merge_results[region_key] = {"error": str(e)}

        print("\n=== Master Database Statistics ===")
        master_stats = master_conn.execute(
            f"""
            SELECT
                COUNT(*) as total_rows,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT plant_name) as unique_plants,
                COUNT(DISTINCT region) as unique_regions,
                COUNT(DISTINCT CONCAT(region, '_', plant_name)) as unique_plant_region_combinations
            FROM {master_table}
        """
        ).fetchone()

        print(f"Master Database: {os.path.basename(master_db_path)}::{master_table}")
        print(f"Total Rows: {master_stats[0]:,}")
        print(f"Date Range: {master_stats[1]} to {master_stats[2]}")
        print(f"Unique Plants: {master_stats[3]}")
        print(f"Unique Regions: {master_stats[4]}")
        print(f"Unique Plant-Region Combinations: {master_stats[5]}")

        print("\n=== Breakdown by Region ===")
        region_breakdown = master_conn.execute(
            f"""
            SELECT
                region,
                COUNT(*) as row_count,
                COUNT(DISTINCT plant_name) as unique_plants,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM {master_table}
            GROUP BY region
            ORDER BY region
        """
        ).fetchall()

        for region, row_count, unique_plants, min_date, max_date in region_breakdown:
            print(f"{region.upper()}:")
            print(f"  Rows: {row_count:,}")
            print(f"  Unique Plants: {unique_plants}")
            print(f"  Date Range: {min_date} to {max_date}")

        print("\n=== Processing Complete ===")
        total_processed = sum(
            r.get("total_rows", 0) for r in merge_results.values() if "error" not in r
        )
        total_inserted = sum(
            r.get("inserted_rows", 0) for r in merge_results.values() if "error" not in r
        )
        total_skipped = sum(
            r.get("skipped_rows", 0) for r in merge_results.values() if "error" not in r
        )
        print(f"Total rows processed from regional databases: {total_processed:,}")
        print(f"Total new rows inserted into master: {total_inserted:,}")
        print(f"Total rows skipped (already existed): {total_skipped:,}")
        print("Master database ready for dashboard consumption!")
    finally:
        master_conn.close()

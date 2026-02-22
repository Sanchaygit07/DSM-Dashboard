from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent.parent  # DSM MASTER


@dataclass(frozen=True)
class SchedulerConfig:
    run_time_local: str  # "HH:MM"
    enabled: bool
    run_rpc: bool
    run_stu: bool


DEFAULT_CONFIG = SchedulerConfig(
    run_time_local="02:00",
    enabled=True,
    run_rpc=True,
    run_stu=True,
)


def _load_config(path: Path) -> SchedulerConfig:
    if not path.exists():
        return DEFAULT_CONFIG
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_CONFIG
    return SchedulerConfig(
        run_time_local=str(raw.get("run_time_local", DEFAULT_CONFIG.run_time_local)),
        enabled=bool(raw.get("enabled", DEFAULT_CONFIG.enabled)),
        run_rpc=bool(raw.get("run_rpc", DEFAULT_CONFIG.run_rpc)),
        run_stu=bool(raw.get("run_stu", DEFAULT_CONFIG.run_stu)),
    )


def _next_run_dt(run_time_local: str, now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now()
    try:
        hh, mm = run_time_local.strip().split(":")
        hh_i, mm_i = int(hh), int(mm)
    except Exception:
        hh_i, mm_i = 2, 0
    target = now.replace(hour=hh_i, minute=mm_i, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target


def _run_script(script_name: str, extra_args: Optional[list[str]] = None) -> int:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"[SCHED] Missing script: {script_path}")
        return 2
    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    print(f"[SCHED] Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=False)
        return int(proc.returncode or 0)
    except Exception as e:
        print(f"[SCHED] Failed to run {script_name}: {e}")
        return 3


def run_ingestion_once(cfg: SchedulerConfig) -> int:
    """Run enabled ingestion steps once. Returns overall exit code."""
    if not cfg.enabled:
        print("[SCHED] Scheduler disabled in config.")
        return 0

    rc_codes: list[int] = []
    if cfg.run_rpc:
        rc_codes.append(_run_script("mastertoduckdb.py"))
    if cfg.run_stu:
        rc_codes.append(_run_script("stu_to_master.py"))

    # Return the worst non-zero code, else 0.
    for c in rc_codes:
        if c != 0:
            return c
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight daily ingestion scheduler (RPC + STU).")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "scheduler_config.json"),
        help="Path to scheduler JSON config file",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run ingestion once immediately and exit",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously and trigger ingestion daily at configured time",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    print(f"[SCHED] Config: {cfg_path} -> {cfg}")

    if args.once:
        raise SystemExit(run_ingestion_once(cfg))

    if not args.loop:
        print("[SCHED] Nothing to do. Use --once or --loop.")
        return

    while True:
        cfg = _load_config(cfg_path)  # reload each cycle so changes apply without restart
        if not cfg.enabled:
            print("[SCHED] Disabled. Sleeping 10 minutes.")
            time.sleep(600)
            continue

        next_dt = _next_run_dt(cfg.run_time_local)
        wait_s = max(1, int((next_dt - datetime.now()).total_seconds()))
        print(f"[SCHED] Next run at {next_dt} (in {wait_s//60}m {wait_s%60}s)")
        time.sleep(wait_s)

        print(f"[SCHED] Triggering ingestion at {datetime.now()}")
        rc = run_ingestion_once(cfg)
        print(f"[SCHED] Ingestion finished rc={rc}")


if __name__ == "__main__":
    main()


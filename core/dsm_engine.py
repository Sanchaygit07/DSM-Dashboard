"""
DSM calculation engine — Deviation Settlement Mechanism formulas and band logic.

Extracted for modular architecture. All formulas and math must remain unchanged.
"""
from __future__ import annotations

import re
import statistics as stats
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Band:
    direction: str        # "UI" | "OI"
    lower_pct: float      # inclusive lower bound
    upper_pct: float      # exclusive upper bound (use 999 for open-ended)
    rate_type: str        # "FLAT" | "PPA_FRAC" | "PPA_MULT" | "SCALED"
    rate_value: float     # flat Rs/kWh OR fraction/multiple 'a' in scaled
    rate_slope: float     # slope 'b' for scaled, else 0
    loss_zone: bool       # True -> goes to OI_Loss (only used when direction="OI")


RATE_FLAT = "FLAT"
RATE_FRAC = "PPA_FRAC"
RATE_MULT = "PPA_MULT"
RATE_SCALED = "SCALED"

MODE_DEFAULT = "DEFAULT"
MODE_DYNAMIC = "DYNAMIC"


def safe_mode(values: List[float]) -> float:
    """MODE with sensible fallback (median) when multimodal/empty."""
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return 0.0
    try:
        return stats.mode(vals)
    except Exception:
        return float(np.median(vals))


def denominator_and_basis(avc: float, sch: float, mode: str, dyn_x: float) -> float:
    """Return denominator (also used as basis for energy) as per rule."""
    if mode == MODE_DYNAMIC:
        return (dyn_x * avc) + ((1.0 - dyn_x) * sch)
    return avc


def direction_from(actual: float, scheduled: float) -> str:
    if actual < scheduled:
        return "UI"
    elif actual > scheduled:
        return "OI"
    return "FLAT"


def slice_pct(abs_err: float, lower: float, upper: float) -> float:
    return max(0.0, min(abs_err, upper) - lower)


def kwh_from_slice(slice_pct_val: float, basis_mw: float) -> float:
    # 15-min block -> 0.25 h; MW -> kW x 1000; energy = P(kw)*h
    return (slice_pct_val / 100.0) * basis_mw * 0.25 * 1000.0


def band_rate(ppa: float, rate_type: str, rate_value: float, rate_slope: float, abs_err: float) -> float:
    if rate_type == RATE_FLAT:
        return rate_value
    if rate_type in (RATE_FRAC, RATE_MULT):
        return rate_value * ppa
    if rate_type == RATE_SCALED:
        return rate_value + rate_slope * abs_err
    return 0.0


def compute_error_pct(df: pd.DataFrame, mode: str, x_pct: float) -> pd.Series:
    """Compute error% column from AvC, Scheduled, Actual per regulation mode."""
    actual = pd.to_numeric(df.get("Actual_MW"), errors="coerce").fillna(0)
    scheduled = pd.to_numeric(df.get("Scheduled_MW"), errors="coerce").fillna(0)
    avc = pd.to_numeric(df.get("AvC_MW"), errors="coerce").replace(0, np.nan)
    num = actual - scheduled
    if str(mode).lower() == "dynamic":
        denom = (float(x_pct) / 100.0) * avc + ((100.0 - float(x_pct)) / 100.0) * scheduled
        denom = denom.replace(0, np.nan)
        return (num / denom * 100.0).fillna(0.0)
    denom = avc
    return (num / denom * 100.0).fillna(0.0)


def compute_basis_mw(df: pd.DataFrame, mode: str, x_pct: float) -> pd.Series:
    """Return the MW basis for energy conversion, aligned with regulation mode."""
    avc = pd.to_numeric(df.get("AvC_MW"), errors="coerce").fillna(0)
    sched = pd.to_numeric(df.get("Scheduled_MW"), errors="coerce").fillna(0)
    if str(mode).lower() == "dynamic":
        return (float(x_pct) / 100.0) * avc + ((100.0 - float(x_pct)) / 100.0) * sched
    return avc


def _normalize_bands_df(bands_df: pd.DataFrame) -> pd.DataFrame:
    """Make bands robust to older saved presets by adding missing columns with defaults."""
    df = bands_df.copy() if isinstance(bands_df, pd.DataFrame) else pd.DataFrame(bands_df or [])
    if "loss_zone" not in df.columns:
        df["loss_zone"] = False
    if "tolerance_cut_pct" not in df.columns:
        df["tolerance_cut_pct"] = 0.0
    if "label" not in df.columns:
        df["label"] = ""
    if "deviated_on" not in df.columns:
        df["deviated_on"] = "AvC"
    for col in ["direction", "lower_pct", "upper_pct", "rate_type", "rate_value", "excess_slope_per_pct"]:
        if col not in df.columns:
            df[col] = 0 if col not in ("direction", "rate_type") else ("UI" if col == "direction" else "flat_per_kwh")
    return df


def parse_bands_from_settings(settings_rows: List[Dict[str, Any]]) -> Tuple[List[Band], pd.DataFrame]:
    """Convert UI bands rows to Band models expected by the engine."""
    type_map = {
        "flat_per_kwh": RATE_FLAT,
        "ppa_fraction": RATE_FRAC,
        "ppa_multiple": RATE_MULT,
        "flat_per_mwh": RATE_FLAT,
        "scaled_excess": RATE_SCALED,
    }
    out: List[Band] = []
    for r in settings_rows or []:
        legacy_type = str(r.get("rate_type", "")).strip().lower()
        mapped_type = type_map.get(legacy_type, RATE_FLAT)
        raw_rate_value = float(r.get("rate_value", 0) or 0)
        rate_value = (raw_rate_value / 1000.0) if legacy_type == "flat_per_mwh" else raw_rate_value
        out.append(Band(
            direction=str(r.get("direction", "")).strip().upper(),
            lower_pct=float(r.get("lower_pct", 0) or 0),
            upper_pct=float(r.get("upper_pct", 0) or 0),
            rate_type=mapped_type,
            rate_value=rate_value,
            rate_slope=float(r.get("excess_slope_per_pct", 0) or 0),
            loss_zone=bool(r.get("loss_zone", False)),
        ))
    out.sort(key=lambda b: (b.direction, b.lower_pct, b.upper_pct))
    bands_df = pd.DataFrame([b.__dict__ for b in out])
    return out, bands_df


def compute_slot_row(slot: Dict[str, Any], bands: List[Band], mode: str, dyn_x: float) -> Dict[str, Any]:
    """Return numeric metrics for a single 15-min slot using the band engine."""
    avc = float(slot["AvC_MW"]) if pd.notna(slot.get("AvC_MW")) else 0.0
    sch = float(slot["Scheduled_MW"]) if pd.notna(slot.get("Scheduled_MW")) else 0.0
    act = float(slot["Actual_MW"]) if pd.notna(slot.get("Actual_MW")) else 0.0
    ppa = float(slot["PPA"]) if pd.notna(slot.get("PPA")) else 0.0

    denom = denominator_and_basis(avc, sch, mode, dyn_x)
    err_pct = 0.0 if denom == 0 else (act - sch) / denom * 100.0
    dirn = direction_from(act, sch)
    abs_err = abs(err_pct)

    ui_dev_kwh = 0.0
    oi_dev_kwh = 0.0
    ui_dsm = 0.0
    oi_dsm = 0.0
    oi_loss = 0.0

    for b in bands:
        if b.direction != dirn:
            continue
        sp = slice_pct(abs_err, b.lower_pct, b.upper_pct)
        if sp <= 0:
            continue
        kwh = kwh_from_slice(sp, denom)
        rate = band_rate(ppa, b.rate_type, b.rate_value, b.rate_slope, abs_err)
        amt = kwh * rate
        if dirn == "UI":
            ui_dev_kwh += kwh
            ui_dsm += amt
        elif dirn == "OI":
            oi_dev_kwh += kwh
            if b.loss_zone:
                oi_loss += amt
            else:
                oi_dsm += amt

    rev_act = act * 0.25 * 1000.0 * ppa
    rev_sch = sch * 0.25 * 1000.0 * ppa
    total_dsm = ui_dsm + oi_dsm
    revenue_loss = total_dsm + oi_loss

    reached = [b for b in bands if b.direction == dirn and abs_err >= b.lower_pct]
    band_level = ""
    if reached:
        top = max(reached, key=lambda x: x.upper_pct)
        lo = int(top.lower_pct)
        up = ("" if top.upper_pct >= 999 else int(top.upper_pct))
        band_level = f"{dirn} {lo}–{up}%" if up != "" else f"{dirn} >{lo}%"

    return {
        "error_pct": err_pct,
        "direction": dirn,
        "abs_err": abs_err,
        "band_level": band_level,
        "UI_Energy_deviation_bands": ui_dev_kwh,
        "OI_Energy_deviation_bands": oi_dev_kwh,
        "Revenue_as_per_generation": rev_act,
        "Scheduled_Revenue_as_per_generation": rev_sch,
        "UI_DSM": ui_dsm,
        "OI_DSM": oi_dsm,
        "OI_Loss": oi_loss,
        "Total_DSM": total_dsm,
        "Revenue_Loss": revenue_loss,
    }


def apply_bands(df: pd.DataFrame, bands: list, unpaid_oi_threshold: float = 15.0) -> pd.DataFrame:
    """Apply band logic to dataframe; add penalty, deviated_kWh, band_label columns."""
    out = df.copy()
    out["direction"] = np.where(out["error_pct"] < 0, "UI", "OI")
    out["abs_err"] = out["error_pct"].abs()

    for col in ["penalty", "deviation_payable", "receivable", "drawl"]:
        out[col] = 0.0

    out["band_label"] = ""
    out["dev_pct"] = 0.0
    out["deviated_kWh"] = 0.0
    out["rate_applied"] = 0.0

    ui_bands = [b for b in bands if b.get("direction") == "UI"]
    oi_bands = [b for b in bands if b.get("direction") == "OI"]

    def tolerance_end(dir_bands: list) -> float:
        if not dir_bands:
            return 0.0
        zero_start = [float(b.get("upper_pct", 0)) for b in dir_bands if float(b.get("lower_pct", 0)) <= 0.0]
        if zero_start:
            return min(zero_start)
        return min(float(b.get("upper_pct", 0)) for b in dir_bands)

    tol_ui = tolerance_end(ui_bands)
    tol_oi = tolerance_end(oi_bands)

    ordered = sorted(bands, key=lambda b: (b.get("direction", ""), float(b.get("lower_pct", 0.0))))
    for b in ordered:
        label = b.get("label", f"{b.get('direction','')}{b.get('lower_pct','')}-{b.get('upper_pct','')}")
        safe = "slice_kWh_" + re.sub(r"[^0-9A-Za-z_]+", "_", label.replace(" ", "_"))
        if safe not in out.columns:
            out[safe] = 0.0

    for b in ordered:
        dirn = b["direction"]
        lower = float(b["lower_pct"])
        upper = float(b["upper_pct"])
        mask_bin = (
            (out["direction"] == dirn) &
            (out["abs_err"] >= lower) &
            (out["abs_err"] < upper)
        )
        out.loc[mask_bin & (out["band_label"] == ""), "band_label"] = b.get("label", "")

    for b in ordered:
        dirn = b["direction"]
        lower = float(b["lower_pct"])
        upper = float(b["upper_pct"])
        tol_end = tol_ui if dirn == "UI" else tol_oi
        eff_lower = np.maximum(lower, tol_end)
        mask_dir = (out["direction"] == dirn)
        if not mask_dir.any():
            continue
        D = out.loc[mask_dir, "abs_err"]
        slice_pct = np.clip(np.minimum(D, upper) - eff_lower, a_min=0.0, a_max=None)
        if not (slice_pct > 0).any():
            continue

        deviated_on = b.get("deviated_on", "AvC")
        rate_type = b["rate_type"]
        rate_value = float(b["rate_value"])
        slope = float(b.get("excess_slope_per_pct", 0.0))
        apply_to = b.get("apply_to", "penalty")
        label = b.get("label", "")

        if "basis_MW" in out.columns:
            basis_series = out.loc[mask_dir, "basis_MW"]
        else:
            basis_series = out.loc[mask_dir, "Scheduled_MW"] if deviated_on == "Scheduled" else out.loc[mask_dir, "AvC_MW"]
        slice_kwh = (slice_pct / 100.0) * basis_series * 0.25 * 1000.0

        if rate_type == "flat_per_kwh":
            rate_series = rate_value
        elif rate_type == "ppa_fraction":
            rate_series = rate_value * pd.to_numeric(out.loc[mask_dir, "PPA"], errors="coerce").fillna(0.0)
        elif rate_type == "ppa_multiple":
            rate_series = rate_value * pd.to_numeric(out.loc[mask_dir, "PPA"], errors="coerce").fillna(0.0)
        elif rate_type == "flat_per_mwh":
            rate_series = rate_value / 1000.0
        elif rate_type == "scaled_excess":
            rate_series = rate_value + slope * D
        else:
            rate_series = 0.0

        amount_series = slice_kwh * rate_series

        if apply_to in ["penalty", "deviation_payable", "receivable", "drawl"]:
            out.loc[mask_dir, apply_to] = out.loc[mask_dir, apply_to] + amount_series

        out.loc[mask_dir, "deviated_kWh"] = out.loc[mask_dir, "deviated_kWh"] + slice_kwh
        out.loc[mask_dir, "dev_pct"] = out.loc[mask_dir, "dev_pct"] + slice_pct

        safe = "slice_kWh_" + re.sub(r"[^0-9A-Za-z_]+", "_", label.replace(" ", "_"))
        out.loc[mask_dir, safe] = out.loc[mask_dir, safe] + slice_kwh

    oi_mask = (out["direction"] == "OI") & (out["abs_err"] >= unpaid_oi_threshold)
    out["actual_kWh_pre"] = out["Actual_MW"] * 0.25 * 1000.0
    out["rev_loss_oi_gt_thresh"] = 0.0
    out.loc[oi_mask, "rev_loss_oi_gt_thresh"] = out.loc[oi_mask, "actual_kWh_pre"] * pd.to_numeric(
        out.loc[oi_mask, "PPA"], errors="coerce"
    ).fillna(0)
    out = out.drop(columns=["actual_kWh_pre"], errors="ignore")

    return out


def summarize(
    df: pd.DataFrame,
    selected_plants: list | None = None,
    bands_rows: List[Dict[str, Any]] | None = None,
    err_mode: str = "default",
    x_pct: float = 50.0,
) -> dict:
    """Build blockwise, plant summary, and kpis from band-applied df."""
    df["Scheduled_MW"] = pd.to_numeric(df["Scheduled_MW"], errors="coerce").fillna(0).clip(lower=0)
    df["Actual_MW"] = pd.to_numeric(df["Actual_MW"], errors="coerce").fillna(0).clip(lower=0)
    df["AvC_MW"] = pd.to_numeric(df["AvC_MW"], errors="coerce").fillna(0).clip(lower=0)
    df["PPA"] = pd.to_numeric(df["PPA"], errors="coerce").fillna(0).clip(lower=0)

    df["sched_kWh"] = df["Scheduled_MW"] * 0.25 * 1000
    df["actual_kWh"] = df["Actual_MW"] * 0.25 * 1000
    df["rev_sched"] = df["sched_kWh"] * df["PPA"]
    df["rev_actual"] = df["actual_kWh"] * df["PPA"]

    df["bin"] = df.get("band_label", "").where(df.get("band_label", "") != "", "Unlabeled")
    if not df.empty:
        blockwise = (
            df.groupby("bin")
            .agg(**{
                "No. of Blocks": ("bin", "size"),
                "Deviated Energy (kWh)": ("deviated_kWh", "sum"),
                "Penalty (₹)": ("penalty", "sum"),
            })
            .reset_index()
        )
    else:
        blockwise = pd.DataFrame({"bin": [], "No. of Blocks": [], "Deviated Energy (kWh)": [], "Penalty (₹)": []})

    bands_list, _ = parse_bands_from_settings(bands_rows or [])
    mode_upper = MODE_DEFAULT if str(err_mode).lower() == "default" else MODE_DYNAMIC
    dyn_x = (float(x_pct) / 100.0) if mode_upper == MODE_DYNAMIC else 0.0

    detail_numeric_rows: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        slot = {
            "region": getattr(r, "region", None),
            "plant_name": getattr(r, "plant_name", getattr(r, "Plant", "")),
            "date": getattr(r, "date", None),
            "time_block": getattr(r, "time_block", getattr(r, "block", None)),
            "from_time": getattr(r, "from_time", None),
            "to_time": getattr(r, "to_time", None),
            "AvC_MW": float(getattr(r, "AvC_MW", 0.0) or 0.0),
            "Scheduled_MW": float(getattr(r, "Scheduled_MW", 0.0) or 0.0),
            "Actual_MW": float(getattr(r, "Actual_MW", 0.0) or 0.0),
            "PPA": float(getattr(r, "PPA", 0.0) or 0.0),
        }
        calc = compute_slot_row(slot, bands_list, mode_upper, dyn_x)
        calc.update({
            "Plant": getattr(r, "Plant", getattr(r, "plant_name", "")),
            "AvC_MW": slot["AvC_MW"],
            "PPA": slot["PPA"],
        })
        detail_numeric_rows.append(calc)
    detail_numeric_df = pd.DataFrame(detail_numeric_rows)

    if not detail_numeric_df.empty and "Plant" in df.columns:
        summaries = []
        for plant, idxs in df.groupby("Plant").groups.items():
            dn = detail_numeric_df.iloc[list(idxs)] if len(detail_numeric_df) == len(df) else detail_numeric_df
            dn_p = dn
            df_p = df.loc[list(idxs)]
            rev_loss_sum = float(dn_p.get("Revenue_Loss", pd.Series(dtype=float)).sum())
            rev_act_sum = float(dn_p.get("Revenue_as_per_generation", pd.Series(dtype=float)).sum())
            actual_kwh_sum = float((df_p["Actual_MW"] * 0.25 * 1000).sum())
            dsm_loss_sum = round(float(dn_p.get("Total_DSM", pd.Series(dtype=float)).sum()), 2)
            loss_pct = round((rev_loss_sum / rev_act_sum * 100.0) if rev_act_sum > 0 else 0.0, 2)
            paise_per_k = round((rev_loss_sum / actual_kwh_sum * 100.0) if actual_kwh_sum > 0 else 0.0, 2)
            cap_mode = round(safe_mode(df_p["AvC_MW"].tolist()), 2)
            ppa_mode = round(safe_mode(df_p["PPA"].tolist()), 2)
            try:
                region_val = str(df_p.get("region").iloc[0]) if "region" in df_p.columns and not df_p.empty else ""
            except Exception:
                region_val = ""
            try:
                data_from = pd.to_datetime(df_p.get("date"), errors="coerce").dt.date.min() if "date" in df_p.columns else None
                data_to = pd.to_datetime(df_p.get("date"), errors="coerce").dt.date.max() if "date" in df_p.columns else None
            except Exception:
                data_from, data_to = None, None
            summaries.append({
                "Region": region_val,
                "Plant name": plant,
                "Plant Capacity": cap_mode,
                "PPA": ppa_mode,
                "Data Available From": data_from,
                "Data Available To": data_to,
                "Revenue Loss (%)": loss_pct,
                "DSM Loss": dsm_loss_sum,
                "Revenue Loss (p/k)": paise_per_k,
            })
        plant_summary = pd.DataFrame(summaries)
    else:
        plant_summary = pd.DataFrame(columns=["Region", "Plant name", "Plant Capacity", "PPA", "Data Available From", "Data Available To", "Revenue Loss (%)", "DSM Loss", "Revenue Loss (p/k)"])

    if selected_plants and len(selected_plants) > 0:
        desired_order = list(dict.fromkeys(selected_plants))
        existing = set(plant_summary["Plant name"]) if not plant_summary.empty else set()
        missing = [p for p in desired_order if p not in existing]
        if missing:
            zeros = pd.DataFrame({
                "Region": [str(df.get("region").iloc[0]) if "region" in df.columns and not df.empty else ""] * len(missing),
                "Plant name": missing,
                "Plant Capacity": [0.0] * len(missing),
                "PPA": [0.0] * len(missing),
                "Data Available From": [None] * len(missing),
                "Data Available To": [None] * len(missing),
                "Revenue Loss (%)": [0.0] * len(missing),
                "DSM Loss": [0.0] * len(missing),
                "Revenue Loss (p/k)": [0.0] * len(missing),
            })
            plant_summary = pd.concat([plant_summary, zeros], ignore_index=True)
        cat = pd.Categorical(plant_summary["Plant name"], categories=desired_order, ordered=True)
        plant_summary = plant_summary.sort_values(by="Plant name", key=lambda s: cat).reset_index(drop=True)

    return {
        "df": df,
        "kpis": [],
        "blockwise": blockwise,
        "plant_summary": plant_summary,
    }


def generate_label(row: dict) -> str:
    """Generate automatic label based on band row values."""
    direction = row.get("direction", "")
    lower_pct = row.get("lower_pct", 0)
    upper_pct = row.get("upper_pct", 0)
    rate_type = row.get("rate_type", "")
    rate_value = row.get("rate_value", 0)

    if upper_pct >= 1000:
        range_str = f"{direction} >{lower_pct}%"
    else:
        range_str = f"{direction} {lower_pct}-{upper_pct}%"

    if float(lower_pct) <= 0.0 and rate_value == 0:
        desc = "(Tolerance)"
    elif rate_type == "flat_per_kwh":
        desc = f"({rate_value} Rs/kWh)"
    elif rate_type == "ppa_fraction":
        desc = f"({rate_value*100:.0f}% of PPA)"
    elif rate_type == "scaled_excess":
        desc = "(scaled)"
    else:
        desc = f"({rate_type})"

    return f"{range_str} {desc}"

"""Flare cohorts and observed SEP labels for the original ten-feature project.

The baseline CSV is never relabelled here.  The extension follows MEMPSEP-III
sections 3--4 (https://arxiv.org/pdf/2310.15390): SWPC C/M/X flares, a six-hour
search, three consecutive five-minute corrected >=10 MeV observations above
both 5 pfu and the maximum three-hour pre-flare background.  Labels are never
inferred from interpolated feature windows, a catalog non-match, or missing data.

The paper does not publish its precise competing-flare grouping code.  We use
the documented deterministic rule below and expose it in the audit, rather than
claiming bit-for-bit reproduction of the original authors' labels.  The original
CSV's fluence normalization is also not documented sufficiently to reproduce it;
standard pfu-second fluences go in the audit, not its sep_fluence_* columns.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .utils import cache_key, ensure_dir, isoformat_z, iter_month_starts, next_month, utc_timestamp

HEK_URL = "https://www.lmsal.com/hek/her"
METHOD_URL = "https://arxiv.org/pdf/2310.15390"
INTEGRAL_COLUMNS = ("GT5", "GT10", "GT30", "GT60", "GT100")
CADENCE = pd.Timedelta(minutes=5)
ASSOCIATION_RULE = "preceding_6h_same_above5_episode_largest_goes_class_v1"


def _date_bounds(start: object, end: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = utc_timestamp(start), utc_timestamp(end)
    if pd.isna(first) or pd.isna(last) or last < first:
        raise ValueError("Flare start/end must be valid dates with start <= end")
    # Date-only end arguments include the entire final day, including Dec 31.
    date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(end)))
    return first, last + (pd.Timedelta(days=1) if date_only else pd.Timedelta(nanoseconds=1))


def goes_class_flux(value: object) -> float:
    """GOES class to peak W/m²; no lexical comparisons between classes."""
    match = re.fullmatch(r"\s*([ABCMX])\s*(\d+(?:\.\d+)?)\s*", str(value).upper())
    if not match:
        return np.nan
    return {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[match[1]] * float(match[2])


def _hek_json(params: dict[str, object], cache_path: Path) -> dict:
    """Cache successful HEK pages only, with bounded retries and pagination."""
    def validate(payload: object) -> dict:
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), list) or "overmax" not in payload:
            raise ValueError(f"Invalid HEK response: {str(payload)[:200]}")
        return payload

    if cache_path.exists():
        try:
            return validate(json.loads(cache_path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            pass
    import requests
    errors = []
    for attempt in range(3):
        try:
            response = requests.get(HEK_URL, params=params, timeout=(20, 120))
            response.raise_for_status()
            payload = validate(response.json())
            ensure_dir(cache_path.parent)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except Exception as exc:
            errors.append(str(exc))
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError("HEK GOES flare query failed: " + "; ".join(errors))


def _number(value: object) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_hek_flares(records: list[dict]) -> pd.DataFrame:
    """Keep the original C/M/X cohort and real reported flare metadata."""
    rows = []
    for record in records:
        if str(record.get("frm_name", "")).upper() != "SWPC":
            continue
        if str(record.get("obs_observatory", "")).upper() != "GOES":
            continue
        onset = utc_timestamp(record.get("event_starttime"))
        class_name = str(record.get("fl_goescls", "")).strip().upper()
        class_flux = goes_class_flux(class_name)
        if pd.isna(onset) or not np.isfinite(class_flux) or class_flux < 1e-6:
            continue
        peak = utc_timestamp(record.get("event_peaktime"))
        end = utc_timestamp(record.get("event_endtime"))
        no_position = str(record.get("noposition", "false")).lower() == "true"
        lon = np.nan if no_position else _number(record.get("event_coord1"))
        lat = np.nan if no_position else _number(record.get("event_coord2"))
        reported_peak = _number(record.get("fl_peakflux"))
        rows.append({
            "FlrOnset": onset, "Flrmaxtime": peak, "Flrendtime": end,
            "event_date": onset.strftime("%m/%d/%y"),
            "start_time": onset, "peak_time": peak, "end_time": end,
            "xrsclass": class_name, "goes_major_class": class_name[0],
            "Peak Flux": reported_peak if np.isfinite(reported_peak) else class_flux,
            "Rise Time": (peak - onset).total_seconds() / 3600 if pd.notna(peak) else np.nan,
            "Duration": (end - onset).total_seconds() / 3600 if pd.notna(end) else np.nan,
            "Fluence": _number(record.get("fl_fluence")),
            "goes_location": str((lon, lat)) if np.isfinite(lon) and np.isfinite(lat) else np.nan,
            "FlrLon": lon, "FlrLat": lat,
            "noaa_active_region": record.get("ar_noaanum"),
            "_flare_id": str(record.get("kb_archivid") or f"HEK-{onset.isoformat()}"),
            "_revision": _number(record.get("revision")),
            "_archived": str(record.get("kb_archivdate", "")),
            "_peak_flux_basis": "reported" if np.isfinite(reported_peak) else "GOES_class",
            "_source_url": HEK_URL,
        })
    if not rows:
        return pd.DataFrame(columns=["FlrOnset", "xrsclass", "Peak Flux", "_flare_id"])
    df = pd.DataFrame(rows)
    # Repeated query boundaries and revisions must not multiply flare samples.
    df = df.sort_values(["_archived", "_revision"], na_position="first")
    df = df.drop_duplicates("_flare_id", keep="last").drop_duplicates("FlrOnset", keep="last")
    return df.sort_values("FlrOnset").reset_index(drop=True)


def fetch_goes_flares(start: object, end: object, cache_dir: str | Path) -> pd.DataFrame:
    """Fetch monthly SWPC/GOES C/M/X flares, including a six-hour boundary halo.

    ``_in_requested_range`` marks the requested cohort.  Halo flares participate
    in attribution across year/month boundaries but are not returned as labelled
    target samples by :func:`label_flare_events`.
    """
    first, stop = _date_bounds(start, end)
    halo_first, halo_stop = first - pd.Timedelta(hours=6), stop + pd.Timedelta(hours=6)
    cache_dir = ensure_dir(cache_dir)
    records: list[dict] = []
    for month in iter_month_starts(halo_first, halo_stop):
        low, high = max(month, halo_first), min(next_month(month), halo_stop)
        print(f"[INFO] GOES flare cohort: {low:%Y-%m-%d} through {high:%Y-%m-%d}", flush=True)
        page, seen_pages = 1, set()
        while True:
            params = {
                "cosec": "2", "cmd": "search", "type": "column", "event_type": "fl",
                "event_starttime": isoformat_z(low), "event_endtime": isoformat_z(high),
                "event_coordsys": "helioprojective", "x1": -5000, "x2": 5000,
                "y1": -5000, "y2": 5000, "param0": "FRM_Name", "op0": "=",
                "value0": "SWPC", "page": page,
            }
            payload = _hek_json(params, cache_dir / f"hek_goes_{cache_key(sorted(params.items()))}.json")
            result = payload["result"]
            signature = cache_key(json.dumps(result, sort_keys=True))
            if signature in seen_pages and result:
                raise RuntimeError(f"HEK repeated page {page} instead of completing {low}..{high}")
            seen_pages.add(signature)
            records.extend(result)
            overmax = str(payload["overmax"]).lower() in {"true", "1"}
            if not overmax:
                break
            if not result or page >= 1000:
                raise RuntimeError(f"HEK pagination did not complete for {low}..{high}")
            page += 1
    df = parse_hek_flares(records)
    df = df[(df["FlrOnset"] >= halo_first) & (df["FlrOnset"] < halo_stop)].copy()
    df["_in_requested_range"] = (df["FlrOnset"] >= first) & (df["FlrOnset"] < stop)
    if not df["_in_requested_range"].any():
        raise RuntimeError("HEK returned no usable SWPC C/M/X flares in the requested period; refusing an empty extension")
    return df.reset_index(drop=True)


def _prepare_flux(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=INTEGRAL_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    df = frame.copy()
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()]
    # Labels use actual observations only: do not round timestamps or interpolate.
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in INTEGRAL_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)
        df[col] = values.where(np.isfinite(values) & (values >= 0) & (values < 1e20))
    return df[list(INTEGRAL_COLUMNS)]


def _three_run(mask: np.ndarray) -> int | None:
    if len(mask) < 3:
        return None
    matches = np.flatnonzero(mask[:-2] & mask[1:-1] & mask[2:])
    return int(matches[0]) if len(matches) else None


def _grid(start: pd.Timestamp, stop: pd.Timestamp) -> pd.DatetimeIndex:
    """Actual five-minute timestamps in the half-open interval [start, stop)."""
    return pd.date_range(start.ceil("5min"), stop.ceil("5min") - CADENCE, freq=CADENCE)


def label_flare_events(
    flares: pd.DataFrame,
    proton_flux_df: pd.DataFrame | Callable | None = None,
    *,
    flux_provider: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None,
    strictcoverage: bool = True,
    end_padding_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label new flares using observed corrected integral proton flux, in pfu.

    The provider accepts UTC ``[start, stop)`` and returns a UTC five-minute
    DatetimeIndex with GT5/GT10/GT30/GT60/GT100 columns.  Requests are bounded to
    a month plus background and event-end padding.  Missing observations never
    become negative labels.  With strict coverage, a label needs every expected
    sample in the three-hour background and six-hour forecast window.  Without
    it, confirmed positives may survive forecast gaps; negatives still require
    complete coverage, and the background always must be complete.

    Competing candidates preceding the earliest enhancement by at most six
    hours and lying in the same continuous above-5-pfu episode are assigned to
    the largest X-ray flare (earlier onset breaks equal-class ties).  The paper
    leaves the exact grouping unspecified; ASSOCIATION_RULE identifies ours.
    """
    if callable(proton_flux_df):
        if flux_provider is not None:
            raise ValueError("Supply only one flux provider")
        flux_provider = proton_flux_df
        proton_flux_df = None
    if flux_provider is None and proton_flux_df is None:
        raise ValueError("Observed corrected integral GOES flux is required")
    if end_padding_days < 1:
        raise ValueError("At least one day of event-end padding is required")
    if flares.empty:
        return flares.copy(), pd.DataFrame()
    work = flares.copy().reset_index(drop=True)
    work["FlrOnset"] = pd.to_datetime(work["FlrOnset"], utc=True, errors="coerce")
    if work["FlrOnset"].isna().any():
        raise ValueError("Flare cohort contains invalid onsets")
    work = work.sort_values("FlrOnset").reset_index(drop=True)
    fixed_flux = _prepare_flux(proton_flux_df) if proton_flux_df is not None else None
    month_cache: dict[str, pd.DataFrame] = {}
    records, observations = [], {}
    for index, flare in work.iterrows():
        onset = flare["FlrOnset"]
        month = onset.normalize().replace(day=1)
        key = month.strftime("%Y-%m")
        if key not in month_cache:
            print(f"[INFO] Observed flare labels: {key}", flush=True)
            low = month - pd.Timedelta(hours=3)
            high = next_month(month) + pd.Timedelta(days=end_padding_days, hours=6)
            month_cache.clear()  # one time block resident at a time
            month_cache[key] = fixed_flux if fixed_flux is not None else _prepare_flux(flux_provider(low, high))
        flux = month_cache[key]
        bg_index = _grid(onset - pd.Timedelta(hours=3), onset)
        post_index = _grid(onset, onset + pd.Timedelta(hours=6))
        background = flux["GT10"].reindex(bg_index)
        forecast = flux["GT10"].reindex(post_index)
        complete_background, complete_forecast = background.notna().all(), forecast.notna().all()
        threshold = max(5.0, float(background.max())) if complete_background else np.nan
        candidate_position = _three_run((forecast > threshold).to_numpy()) if complete_background else None
        candidate = post_index[candidate_position] if candidate_position is not None else pd.NaT
        status = "candidate" if pd.notna(candidate) else "negative"
        if not complete_background or (not complete_forecast and (strictcoverage or pd.isna(candidate))):
            status = "excluded_missing_label_flux"
            candidate = pd.NaT
        actual_peak = _number(flare.get("Peak Flux"))
        if not np.isfinite(actual_peak):
            actual_peak = goes_class_flux(flare.get("xrsclass"))
        if not np.isfinite(actual_peak):
            status, candidate = "excluded_missing_xray_class", pd.NaT
        event_end = pd.NaT
        if pd.notna(candidate):
            end_index = pd.date_range(candidate, min(flux.index.max(), candidate + pd.Timedelta(days=end_padding_days)), freq=CADENCE)
            values = flux["GT10"].reindex(end_index)
            end_position = _three_run((values < 5.0).to_numpy())
            if end_position is not None and values.iloc[:end_position + 3].notna().all():
                event_end = end_index[end_position]
        records.append({
            "_row": index, "flare_id": flare.get("_flare_id", f"flare-{index}"),
            "FlrOnset": onset, "xrsclass": flare.get("xrsclass"),
            "in_requested_range": bool(flare.get("_in_requested_range", True)),
            "status": status, "event_type": np.nan,
            "background_max_gt10_pfu": background.max() if background.notna().any() else np.nan,
            "background_valid_fraction": float(background.notna().mean()),
            "forecast_valid_fraction": float(forecast.notna().mean()),
            "candidate_sep_start": candidate, "observed_sep_end": event_end,
            "peak_xray_w_m2": actual_peak,
            "peak_xray_basis": flare.get("_peak_flux_basis", "GOES_class"),
            "source_url": flare.get("_source_url", HEK_URL),
            # Full download records belong in the build manifest, not repeated
            # once per flare (the source store accumulates these over years).
            "proton_source_provenance": json.dumps({key: value for key, value in flux.attrs.items() if key != "source_records"}, sort_keys=True, default=str),
            "method_url": METHOD_URL, "association_rule": ASSOCIATION_RULE,
            "competing_winner_id": "", "fluence_legacy_convention": "undocumented_not_populated",
        })
        # Save only the small slices needed by candidate attribution and metrics.
        metric_start = candidate if pd.notna(candidate) else onset.ceil("5min")
        metric_stop = event_end if pd.notna(candidate) else onset + pd.Timedelta(hours=6)
        observations[index] = flux.reindex(_grid(metric_start, metric_stop)).copy() if pd.notna(metric_stop) else None
        if pd.notna(candidate):
            observations[(index, "negative")] = flux.reindex(post_index).copy()

    audit = pd.DataFrame(records)
    candidates = audit[audit.status.eq("candidate")].sort_values(["candidate_sep_start", "FlrOnset"])
    unassigned = set(candidates.index)
    for seed_index, seed in candidates.iterrows():
        if seed_index not in unassigned:
            continue
        first_onset = seed["candidate_sep_start"]
        episode_end = seed["observed_sep_end"]
        # An unresolved long episode is bounded to the six-hour association horizon.
        limit = episode_end if pd.notna(episode_end) else first_onset + pd.Timedelta(hours=6)
        eligible = candidates.loc[list(unassigned)]
        eligible = eligible[
            (eligible.FlrOnset <= first_onset)
            & (eligible.FlrOnset >= first_onset - pd.Timedelta(hours=6))
            & (eligible.candidate_sep_start < limit)
        ]
        winner = eligible.sort_values(["peak_xray_w_m2", "FlrOnset"], ascending=[False, True]).index[0]
        for member in eligible.index:
            audit.loc[member, "event_type"] = int(member == winner)
            audit.loc[member, "status"] = "positive" if member == winner else "negative_competing_flare"
            audit.loc[member, "competing_winner_id"] = str(audit.loc[winner, "flare_id"])
            unassigned.remove(member)
    audit.loc[audit.status.eq("negative"), "event_type"] = 0

    labels = []
    for index, row in audit.iterrows():
        if not row["in_requested_range"]:
            continue
        if pd.isna(row["event_type"]):
            continue
        flare = work.iloc[int(row["_row"])].to_dict()
        label = int(row["event_type"])
        flare["event_type"] = label
        if label:
            flare["sep_start"], flare["sep_end"] = row["candidate_sep_start"], row["observed_sep_end"]
            flare["sep_rise"] = (flare["sep_start"] - flare["FlrOnset"]).total_seconds() / 3600
            flare["sep_duration"] = (flare["sep_end"] - flare["sep_start"]).total_seconds() / 3600 if pd.notna(flare["sep_end"]) else np.nan
            metric_frame = observations[index]
        else:
            flare["sep_start"], flare["sep_end"] = flare["FlrOnset"], flare["FlrOnset"] + pd.Timedelta(hours=6)
            flare["sep_rise"], flare["sep_duration"] = -9999.0, -9999.0
            metric_frame = observations.get((index, "negative"), observations[index])
        for band, col in enumerate(INTEGRAL_COLUMNS, start=1):
            # Missing event ends cannot support whole-event peak or fluence claims.
            complete = metric_frame is not None and not metric_frame.empty and metric_frame[col].notna().all()
            if complete:
                expected = pd.date_range(metric_frame.index.min(), metric_frame.index.max(), freq=CADENCE)
                complete = len(expected) == len(metric_frame)
            flare[f"sep_peak_{band}"] = float(metric_frame[col].max()) if complete else np.nan
            flare[f"sep_fluence_{band}"] = np.nan
            audit.loc[index, f"{col}_metric_valid_fraction"] = float(metric_frame[col].notna().mean()) if metric_frame is not None and not metric_frame.empty else 0.0
            audit.loc[index, f"{col}_fluence_pfu_seconds"] = float(metric_frame[col].sum() * 300) if complete else np.nan
        labels.append({key: value for key, value in flare.items() if not key.startswith("_")})
    audit = audit[audit.in_requested_range].drop(columns="_row").reset_index(drop=True)
    labels_df = pd.DataFrame(labels)
    if not labels_df.empty:
        labels_df = labels_df.sort_values("FlrOnset").reset_index(drop=True)
    print(f"[INFO] Extension labels: {len(labels_df)} resolved flares; {audit.event_type.isna().sum()} excluded for unresolved observations", flush=True)
    return labels_df, audit

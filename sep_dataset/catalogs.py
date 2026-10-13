from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import (
    cache_key,
    compact_spaces,
    ensure_dir,
    http_get_text,
    parse_float,
    parse_month_day_hm,
    parse_ymd_hm,
    strip_html_to_text,
    utc_timestamp,
)


NOAA_SPE_URLS = [
    "https://ftp.swpc.noaa.gov/pub/indices/SPE.txt",
    "https://umbra.nascom.nasa.gov/SEP/seps.html",
]

CDAW_MAJOR_SEP_URL = "https://cdaw.gsfc.nasa.gov/CME_list/sepe/"

DONKI_ENDPOINTS = [
    "https://ccmc.gsfc.nasa.gov/DONKI-API/get/SEP",
    "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/SEP",
    "https://api.nasa.gov/DONKI/SEP",
]


EVENT_COLUMNS = [
    "event_id",
    "source",
    "source_url",
    "start_time",
    "max_time",
    "end_time",
    "peak_pfu_gt10",
    "flare_time",
    "flare_class",
    "flare_location",
    "active_region",
    "cme_time",
    "cme_speed_kms",
    "cme_width_deg",
    "gle",
    "instruments",
    "notes",
]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _finalize_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty_events()
    df = pd.DataFrame(rows)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["start_time", "max_time", "end_time", "flare_time", "cme_time"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ["peak_pfu_gt10", "cme_speed_kms", "cme_width_deg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start_time"]).copy()
    df = df.sort_values("start_time").reset_index(drop=True)
    return df[EVENT_COLUMNS]


def _split_noaa_cells(line: str) -> list[str]:
    if "|" in line:
        return [compact_spaces(cell) for cell in line.split("|") if compact_spaces(cell)]
    return [compact_spaces(cell) for cell in re.split(r"\s{2,}", line) if compact_spaces(cell)]


def parse_noaa_spe_events(text: str, source_url: str = "") -> pd.DataFrame:
    """Parse the NOAA/SWPC Solar Proton Events list.

    The source has appeared as fixed-width text, an HTML table, and a PRE block
    at different mirrors. This parser accepts all three by using the date cells
    rather than fragile column widths.
    """
    text = strip_html_to_text(text)
    rows: list[dict[str, object]] = []
    current_year: int | None = None
    month_names = "|".join(name.title() for name in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    start_re = re.compile(rf"^({month_names})\s+\d{{1,2}}/\d{{3,4}}\b")

    for raw_line in text.splitlines():
        line = compact_spaces(raw_line)
        if not line:
            continue
        year_cells = re.findall(r"\b(19[7-9]\d|20\d{2})\b", line)
        if year_cells and not start_re.search(line):
            current_year = int(year_cells[0])
            continue
        if current_year is None or not start_re.search(line):
            continue

        cells = _split_noaa_cells(line)
        if len(cells) < 3:
            continue
        start_time = parse_month_day_hm(cells[0], current_year)
        max_time = parse_month_day_hm(cells[1], current_year, reference=start_time)
        if pd.isna(start_time):
            continue
        peak = parse_float(cells[2])

        flare_time = pd.NaT
        flare_class = ""
        flare_location = ""
        active_region = ""
        if len(cells) >= 6:
            flare_time = parse_month_day_hm(cells[5], current_year, reference=start_time)
        if len(cells) >= 7:
            flare_class = cells[6]
        if len(cells) >= 8:
            flare_location = cells[7]
        if len(cells) >= 9:
            active_region = cells[8]

        rows.append(
            {
                "event_id": f"NOAA-SPE-{start_time.strftime('%Y%m%dT%H%M')}",
                "source": "NOAA_SWPC_SPE",
                "source_url": source_url,
                "start_time": start_time,
                "max_time": max_time,
                "peak_pfu_gt10": peak,
                "flare_time": flare_time,
                "flare_class": flare_class,
                "flare_location": flare_location,
                "active_region": active_region,
            }
        )
    return _finalize_events(rows)


def fetch_noaa_spe_events(cache_dir: str | Path = "data/cache/catalogs") -> pd.DataFrame:
    cache_dir = ensure_dir(cache_dir)
    errors: list[str] = []
    for url in NOAA_SPE_URLS:
        try:
            cache_path = cache_dir / f"noaa_spe_{cache_key(url)}.txt"
            text = http_get_text(url, cache_path=cache_path)
            events = parse_noaa_spe_events(text, source_url=url)
            if not events.empty:
                return events
            errors.append(f"{url}: no parseable rows")
        except Exception as exc:  # pragma: no cover - live network only
            errors.append(f"{url}: {exc}")
    raise RuntimeError("NOAA SPE catalog unavailable: " + "; ".join(errors))


def _parse_cdaw_month_day(value: str, year: int, reference: pd.Timestamp | None = None) -> pd.Timestamp:
    if "/" not in value:
        return pd.NaT
    return parse_month_day_hm(value.replace("/", " "), year, reference=reference)


def parse_cdaw_major_sep_events(text: str, source_url: str = CDAW_MAJOR_SEP_URL) -> pd.DataFrame:
    """Parse the CDAW major SEP event table."""
    text = strip_html_to_text(text)
    rows: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        line = compact_spaces(raw_line)
        if not re.match(r"^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}\b", line):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        start_time = parse_ymd_hm(parts[0], parts[1])
        max_time = parse_ymd_hm(parts[2], parts[3])
        peak = parse_float(parts[4])
        year = int(parts[0][:4])

        sep_peak_time = pd.NaT
        sep_peak = np.nan
        gle = ""
        cme_time = pd.NaT
        cme_speed = np.nan
        cme_width = np.nan
        flare_time = pd.NaT
        flare_class = ""
        flare_location = ""

        if len(parts) >= 9:
            sep_peak_time = _parse_cdaw_month_day(parts[6], year, reference=start_time)
            if pd.notna(sep_peak_time) and len(parts) >= 8:
                hhmm = parts[7]
                sep_peak_time = pd.Timestamp(
                    year=sep_peak_time.year,
                    month=sep_peak_time.month,
                    day=sep_peak_time.day,
                    hour=int(hhmm.split(":")[0]),
                    minute=int(hhmm.split(":")[1]),
                    tz="UTC",
                )
            sep_peak = parse_float(parts[8])
        if len(parts) >= 10:
            gle = parts[9]
        if len(parts) >= 12 and re.match(r"^\d{4}/\d{2}/\d{2}$", parts[10]):
            cme_time = parse_ymd_hm(parts[10], parts[11])
        if len(parts) >= 13:
            cme_speed = parse_float(parts[12])
        if len(parts) >= 14:
            cme_width = parse_float(parts[13])
        if len(parts) >= 15 and re.match(r"^[NS]\d{2}[EW]\d{2,3}$|^[<>]?[EW]\d{2,3}$", parts[14]):
            flare_location = parts[14]
        if len(parts) >= 17:
            flare_time = _parse_cdaw_month_day(parts[15], year, reference=start_time)
            if pd.notna(flare_time) and ":" in parts[16]:
                flare_time = pd.Timestamp(
                    year=flare_time.year,
                    month=flare_time.month,
                    day=flare_time.day,
                    hour=int(parts[16].split(":")[0]),
                    minute=int(parts[16].split(":")[1]),
                    tz="UTC",
                )
        if len(parts) >= 20:
            flare_class = parts[19]

        rows.append(
            {
                "event_id": f"CDAW-SEP-{start_time.strftime('%Y%m%dT%H%M')}",
                "source": "CDAW_MAJOR_SEP",
                "source_url": source_url,
                "start_time": start_time,
                "max_time": max_time if pd.notna(max_time) else sep_peak_time,
                "peak_pfu_gt10": peak if np.isfinite(peak) else sep_peak,
                "flare_time": flare_time,
                "flare_class": flare_class,
                "flare_location": flare_location,
                "cme_time": cme_time,
                "cme_speed_kms": cme_speed,
                "cme_width_deg": cme_width,
                "gle": gle,
                "notes": " ".join(parts[20:]) if len(parts) > 20 else "",
            }
        )
    return _finalize_events(rows)


def fetch_cdaw_major_sep_events(cache_dir: str | Path = "data/cache/catalogs") -> pd.DataFrame:
    cache_dir = ensure_dir(cache_dir)
    cache_path = cache_dir / "cdaw_major_sep.html"
    text = http_get_text(CDAW_MAJOR_SEP_URL, cache_path=cache_path)
    return parse_cdaw_major_sep_events(text, source_url=CDAW_MAJOR_SEP_URL)


def _download_json(url: str, params: dict[str, object], cache_path: Path) -> object:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    import requests

    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    ensure_dir(cache_path.parent)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _linked_event_summary(item: dict[str, object]) -> str:
    linked = item.get("linkedEvents") or item.get("linked_events") or []
    if not isinstance(linked, list):
        return ""
    labels = []
    for entry in linked:
        if isinstance(entry, dict):
            labels.append(str(entry.get("activityID") or entry.get("eventID") or entry.get("id") or entry))
        else:
            labels.append(str(entry))
    return ";".join(labels)


def parse_donki_sep_events(payload: object, source_url: str = "") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    if not isinstance(payload, list):
        return _empty_events()

    for item in payload:
        if not isinstance(item, dict):
            continue
        start = (
            item.get("eventTime")
            or item.get("startTime")
            or item.get("beginTime")
            or item.get("sepStartTime")
            or item.get("activityStartTime")
        )
        start_time = utc_timestamp(start)
        if pd.isna(start_time):
            continue
        instruments = item.get("instruments") or item.get("instrument") or ""
        if isinstance(instruments, list):
            instruments = ";".join(str(x.get("displayName", x)) if isinstance(x, dict) else str(x) for x in instruments)

        rows.append(
            {
                "event_id": str(item.get("sepID") or item.get("activityID") or f"DONKI-SEP-{start_time.strftime('%Y%m%dT%H%M')}"),
                "source": "NASA_DONKI_SEP",
                "source_url": source_url,
                "start_time": start_time,
                "instruments": instruments,
                "notes": compact_spaces(str(item.get("note") or item.get("notes") or "")),
                "cme_time": utc_timestamp(item.get("cmeTime")) if item.get("cmeTime") else pd.NaT,
                "flare_time": utc_timestamp(item.get("flareTime")) if item.get("flareTime") else pd.NaT,
                "active_region": item.get("activeRegionNum") or item.get("activeRegion") or "",
                "linked_events": _linked_event_summary(item),
            }
        )
    return _finalize_events(rows)


def fetch_donki_sep_events(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path = "data/cache/catalogs",
    api_key: str = "DEMO_KEY",
) -> pd.DataFrame:
    cache_dir = ensure_dir(cache_dir)
    start_s = utc_timestamp(start).strftime("%Y-%m-%d")
    end_s = utc_timestamp(end).strftime("%Y-%m-%d")
    errors: list[str] = []
    for endpoint in DONKI_ENDPOINTS:
        params: dict[str, object] = {"startDate": start_s, "endDate": end_s}
        if "api.nasa.gov" in endpoint:
            params["api_key"] = api_key
        try:
            payload = _download_json(endpoint, params, cache_dir / f"donki_sep_{cache_key(endpoint, start_s, end_s, api_key)}.json")
            events = parse_donki_sep_events(payload, source_url=endpoint)
            if not events.empty:
                return events
            errors.append(f"{endpoint}: no rows")
        except Exception as exc:  # pragma: no cover - live network only
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("DONKI SEP catalog unavailable: " + "; ".join(errors))


def load_memsep_events(path_or_url: str | Path) -> pd.DataFrame:
    path_or_url = str(path_or_url)
    df = pd.read_csv(path_or_url)
    if "event_type" in df.columns:
        df = df[pd.to_numeric(df["event_type"], errors="coerce").fillna(0).astype(int) == 1].copy()
    rows: list[dict[str, object]] = []
    for i, row in df.iterrows():
        start = None
        for col in ["SEPStart", "sep_start", "SEP_start", "event_start", "FlrOnset", "flare_onset"]:
            if col in row and pd.notna(row[col]):
                start = row[col]
                break
        start_time = utc_timestamp(start)
        if pd.isna(start_time):
            continue
        flare_time = utc_timestamp(row.get("FlrOnset")) if "FlrOnset" in row else pd.NaT
        rows.append(
            {
                "event_id": f"MEMSEP-{i}",
                "source": "MEMSEP_LOCAL",
                "source_url": path_or_url,
                "start_time": start_time,
                "flare_time": flare_time,
                "peak_pfu_gt10": parse_float(row.get("sep_peak_2", np.nan)),
                "flare_class": row.get("fl_goescls", row.get("flare_class", "")),
                "flare_location": row.get("fl_location", row.get("flare_location", "")),
                "active_region": row.get("ar_noaanum", row.get("active_region", "")),
            }
        )
    return _finalize_events(rows)


def filter_events(df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    start_ts = utc_timestamp(start)
    end_ts = utc_timestamp(end)
    return df[(df["start_time"] >= start_ts) & (df["start_time"] <= end_ts)].copy()


def merge_event_catalogs(catalogs: Iterable[pd.DataFrame], tolerance_hours: float = 6.0) -> pd.DataFrame:
    frames = [df for df in catalogs if df is not None and not df.empty]
    if not frames:
        return _empty_events()
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
    tolerance = pd.Timedelta(hours=tolerance_hours)
    clusters: list[list[int]] = []
    current: list[int] = []
    current_ref: pd.Timestamp | None = None

    for idx, row in raw.iterrows():
        start_time = row["start_time"]
        if current_ref is None or start_time - current_ref <= tolerance:
            current.append(idx)
            current_ref = start_time if current_ref is None else min(current_ref, start_time)
        else:
            clusters.append(current)
            current = [idx]
            current_ref = start_time
    if current:
        clusters.append(current)

    merged_rows: list[dict[str, object]] = []
    for cluster_id, indexes in enumerate(clusters, start=1):
        group = raw.loc[indexes].copy()
        row: dict[str, object] = {}
        row["event_id"] = f"SEP-{cluster_id:04d}-{group['start_time'].min().strftime('%Y%m%dT%H%M')}"
        row["source"] = "+".join(sorted(set(group["source"].dropna().astype(str))))
        row["source_url"] = ";".join(sorted(set(group["source_url"].dropna().astype(str))))
        row["start_time"] = group["start_time"].min()
        row["max_time"] = group["max_time"].dropna().iloc[0] if group["max_time"].notna().any() else pd.NaT
        row["end_time"] = group["end_time"].dropna().iloc[0] if group["end_time"].notna().any() else pd.NaT
        row["peak_pfu_gt10"] = group["peak_pfu_gt10"].max(skipna=True)
        for col in ["flare_time", "cme_time"]:
            row[col] = group[col].dropna().iloc[0] if col in group and group[col].notna().any() else pd.NaT
        for col in ["flare_class", "flare_location", "active_region", "gle", "instruments", "notes"]:
            values = [compact_spaces(v) for v in group.get(col, pd.Series(dtype=object)).dropna().astype(str) if compact_spaces(v)]
            row[col] = ";".join(dict.fromkeys(values))
        for col in ["cme_speed_kms", "cme_width_deg"]:
            row[col] = pd.to_numeric(group.get(col, pd.Series(dtype=float)), errors="coerce").max(skipna=True)
        row["catalog_members"] = ";".join(group["event_id"].dropna().astype(str))
        merged_rows.append(row)

    merged = _finalize_events(merged_rows)
    extra = pd.DataFrame(merged_rows)
    if "catalog_members" in extra.columns:
        merged["catalog_members"] = extra["catalog_members"].values
    return merged


def collect_event_catalogs(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path = "data/cache/catalogs",
    memsep_csv: str | Path | None = None,
    include_noaa: bool = True,
    include_cdaw: bool = True,
    include_donki: bool = True,
    donki_api_key: str = "DEMO_KEY",
    merge_tolerance_hours: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalogs: list[pd.DataFrame] = []
    if include_noaa:
        catalogs.append(filter_events(fetch_noaa_spe_events(cache_dir), start, end))
    if include_cdaw:
        catalogs.append(filter_events(fetch_cdaw_major_sep_events(cache_dir), start, end))
    if include_donki:
        catalogs.append(filter_events(fetch_donki_sep_events(start, end, cache_dir, api_key=donki_api_key), start, end))
    if memsep_csv:
        catalogs.append(filter_events(load_memsep_events(memsep_csv), start, end))
    raw = pd.concat([df for df in catalogs if not df.empty], ignore_index=True, sort=False) if catalogs else _empty_events()
    merged = merge_event_catalogs(catalogs, tolerance_hours=merge_tolerance_hours)
    return raw, merged

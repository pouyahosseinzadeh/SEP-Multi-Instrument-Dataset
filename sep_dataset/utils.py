from __future__ import annotations

import hashlib
import html
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_key(*parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert("UTC")


def isoformat_z(ts: pd.Timestamp) -> str:
    ts = utc_timestamp(ts)
    if pd.isna(ts):
        raise ValueError("cannot format NaT")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    ts = utc_timestamp(ts)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def next_month(ts: pd.Timestamp) -> pd.Timestamp:
    ts = month_start(ts)
    if ts.month == 12:
        return pd.Timestamp(year=ts.year + 1, month=1, day=1, tz="UTC")
    return pd.Timestamp(year=ts.year, month=ts.month + 1, day=1, tz="UTC")


def iter_month_starts(start: pd.Timestamp, stop: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cur = month_start(start)
    stop = utc_timestamp(stop)
    while cur < stop:
        yield cur
        cur = next_month(cur)


def strip_html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?i)</t[dh]>\s*<t[dh][^>]*>", " | ", text)
    text = re.sub(r"(?i)</tr\s*>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    return text


def compact_spaces(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def parse_float(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).replace(",", "").strip()
    if not text or text in {"--", "---", "----", "-----", "n/a", "NA"}:
        return np.nan
    if text.upper() == "H":
        return 360.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def http_get_text(
    url: str,
    params: dict[str, object] | None = None,
    timeout: int = 60,
    retries: int = 3,
    cache_path: str | Path | None = None,
) -> str:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            import requests

            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            text = response.text
            if cache_path is not None:
                ensure_dir(cache_path.parent)
                cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:  # pragma: no cover - exercised by live network runs
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def clean_numeric_series(series: pd.Series, fill_values: Iterable[object] = ()) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    fills = {parse_float(v) for v in fill_values}
    fills.update({-9999.0, 9999.0, 99999.0, 99999.9, 99999.8984375, 1e30, -1e30, 1e31, -1e31})
    for fill in fills:
        if np.isfinite(fill):
            out[np.isclose(out, fill, rtol=0, atol=max(abs(fill) * 1e-7, 1e-9))] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def parse_month_day_hm(value: object, year: int, reference: pd.Timestamp | None = None) -> pd.Timestamp:
    text = compact_spaces(value)
    if not text or "--" in text:
        return pd.NaT
    match = re.search(
        r"(?P<month>[A-Za-z]{3}|\d{1,2})\s*/?\s*(?P<day>\d{1,2})[/\s]+(?P<hour>\d{1,2})(?::?(?P<minute>\d{2}))",
        text,
    )
    if not match:
        return pd.NaT
    month_text = match.group("month")
    month = int(month_text) if month_text.isdigit() else MONTHS.get(month_text[:3].lower())
    if not month:
        return pd.NaT
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    extra_day = 0
    if hour == 24:
        hour = 0
        extra_day = 1
    ts = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, tz="UTC")
    if extra_day:
        ts += pd.Timedelta(days=extra_day)
    if reference is not None and ts < reference - pd.Timedelta(days=180):
        ts = pd.Timestamp(year=year + 1, month=month, day=day, hour=hour, minute=minute, tz="UTC")
    return ts


def parse_ymd_hm(date_value: object, time_value: object) -> pd.Timestamp:
    text = f"{date_value} {time_value}".replace("/", "-")
    return utc_timestamp(text)

"""Native SMARP/SHARP keyword observations for two photospheric predictors.

No magnetogram segments, regressed SMHARP substitutes, linear interpolation,
median filling, or unrelated active regions are used. Five-minute outputs are
backward holds of native 96/12-minute observations, with bounded ages and masks.
Definitive products are retrospective; this is not a reconstruction of their
real-time publication latency. Native calibration differences are not removed.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests
from astropy.time import Time

FEATURES = ("USFLUXL", "R_VALUE")
JSOC_URL = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info"
SHARP_START = pd.Timestamp("2010-05-01T00:00:00Z")
SMARP_START = pd.Timestamp("1996-04-23T00:00:00Z")
QUALITY_REFERENCE = "https://github.com/mbobra/SMARPs/blob/main/paper/Matching_TARPs_and_HARPs.ipynb"
PARAMETER_REFERENCE = "https://doi.org/10.3847/1538-4365/ac1f1d"


class MagneticSourceError(RuntimeError):
    pass


class MagneticQueryError(MagneticSourceError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f"JSOC status {status}: {message}")


@dataclass(frozen=True)
class NativeSpec:
    era: str
    series: str
    patch_key: str
    satellite: str
    cadence_minutes: int
    quality_limit: int
    integration_guard_minutes: int


SMARP = NativeSpec("SMARP", "mdi.smarp_cea_96m", "TARPNUM", "SOHO/MDI", 96, 262144, 5)
SHARP = NativeSpec("SHARP", "hmi.sharp_cea_720s", "HARPNUM", "SDO/HMI", 12, 65536, 12)
SPECS = {spec.series: spec for spec in (SMARP, SHARP)}


def _utc(value) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize("UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")


def parse_jsoc_times(values) -> pd.DatetimeIndex:
    """Convert explicit native TAI strings using Astropy's leap-second table."""
    result = pd.Series(pd.NaT, index=np.arange(len(values)), dtype="datetime64[ns, UTC]")
    tai_positions, tai_values = [], []
    for position, value in enumerate(values):
        text = str(value).strip()
        match = re.fullmatch(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2}:\d{2}:\d{2}(?:\.\d+)?)_TAI", text)
        if match:
            tai_positions.append(position)
            tai_values.append(f"{match[1]}-{match[2]}-{match[3]}T{match[4]}")
        elif text.endswith("Z") or re.search(r"[+-]\d\d:\d\d$", text):
            result.iloc[position] = pd.to_datetime(text, utc=True, errors="coerce")
        # Bare local/TAI-looking times are not silently interpreted as UTC.
    if tai_values:
        utc_times = Time(tai_values, format="isot", scale="tai").utc
        utc_times.precision = 9
        converted = pd.to_datetime(utc_times.isot, utc=True)
        result.iloc[tai_positions] = converted
    return pd.DatetimeIndex(result)


def _quality(value) -> int | None:
    try:
        text = str(value).strip()
        if text.lower().startswith("0x"):
            return int(text, 16)
        number = float(text)
        return int(number) if np.isfinite(number) and number.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def _ar_number(value) -> int | None:
    try:
        number = float(value)
        return int(number) if np.isfinite(number) and number > 0 and number.is_integer() else None
    except (ValueError, TypeError):
        return None


def _members(value, primary) -> tuple[int, ...]:
    members = {int(token) for token in re.findall(r"-?\d+", str(value)) if int(token) > 0}
    primary = _ar_number(primary)
    if primary is not None:
        members.add(primary)
    return tuple(sorted(members))


def parse_keyword_payload(payload: dict, spec: NativeSpec) -> pd.DataFrame:
    if not isinstance(payload, dict) or payload.get("status") != 0:
        raise MagneticSourceError(f"JSOC {spec.series}: {str(payload)[:250]}")
    keywords = payload.get("keywords")
    if not isinstance(keywords, list):
        raise MagneticSourceError(f"JSOC {spec.series} lacks keyword arrays")
    columns = {entry.get("name"): entry.get("values") for entry in keywords if isinstance(entry, dict)}
    required = {"T_REC", "T_OBS", spec.patch_key, "NOAA_AR", "NOAA_ARS", "QUALITY", *FEATURES}
    if payload.get("count") == 0 and keywords == []:
        return pd.DataFrame(columns=sorted(required))
    if not required.issubset(columns):
        raise MagneticSourceError(f"JSOC {spec.series} missing keywords: {sorted(required-set(columns))}")
    sizes = {len(value) for value in columns.values() if isinstance(value, list)}
    if len(sizes) != 1 or any(not isinstance(value, list) for value in columns.values()):
        raise MagneticSourceError("JSOC keyword arrays have inconsistent lengths")
    count = next(iter(sizes))
    if payload.get("count") != count:
        raise MagneticSourceError("JSOC count disagrees with keyword arrays; refusing truncated data")
    return pd.DataFrame(columns)


def normalize_native(frame: pd.DataFrame, spec: NativeSpec) -> pd.DataFrame:
    required = {"T_REC", "T_OBS", spec.patch_key, "NOAA_AR", "NOAA_ARS", "QUALITY", *FEATURES}
    if not required.issubset(frame.columns):
        raise MagneticSourceError(f"Native CSV lacks {sorted(required-set(frame.columns))}")
    data = frame.copy()
    data["record_time"] = parse_jsoc_times(data.T_REC.tolist())
    data["observation_time"] = parse_jsoc_times(data.T_OBS.tolist())
    # MDI's T_OBS can be later than its nominal slot. Neither clock may look
    # ahead; HMI's 1350-second averaging window gets a conservative 12m guard.
    clock = pd.concat([data.record_time, data.observation_time], axis=1).max(axis=1)
    data["available_at"] = clock + pd.Timedelta(minutes=spec.integration_guard_minutes)
    data["native_quality"] = data.QUALITY.map(_quality)
    data["quality_ok"] = data.native_quality.map(lambda value: value is not None and pd.notna(value) and 0 <= value < spec.quality_limit)
    data["members"] = [_members(value, primary) for value, primary in zip(data.NOAA_ARS, data.NOAA_AR)]
    data["patch"] = data[spec.patch_key].astype(str)
    data["series"] = spec.series
    for feature in FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        # Zero R_VALUE is legitimate and must not become missing.
        data[feature] = values.where(np.isfinite(values) & (values >= 0) & (values < 1e38))
    data = data[data.record_time.notna() & data.observation_time.notna()].copy()
    if len(frame) and data.empty:
        raise MagneticSourceError("All native timestamps are invalid or lack an explicit UTC/TAI scale")
    return data.sort_values("available_at").reset_index(drop=True)


class MagneticStore:
    """Monthly native keyword cache with a per-NOAA-region lookup index.

    native_csv may be a raw-keyword CSV with a required ``series`` column, or
    a directory containing ``SMARP_YYYY-MM.csv`` / ``SHARP_YYYY-MM.csv``. Required
    native columns are listed in metadata(). No reconstructed SMHARP table is
    silently accepted. Network failures raise; missing records remain missing.
    """

    def __init__(self, cache_dir, progress=print, *, native_csv=None,
                 max_age_cadences: float = 2.0, memory_months: int = 6,
                 allow_multi_region_patch: bool = False, timeout: int = 180,
                 retries: int = 3, integration_guards: dict | None = None):
        if max_age_cadences <= 0:
            raise ValueError("max_age_cadences must be positive")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.progress, self.native_csv = progress, Path(native_csv).expanduser() if native_csv else None
        self.max_age_cadences, self.memory_months = float(max_age_cadences), max(2, int(memory_months))
        self.allow_multi_region_patch = bool(allow_multi_region_patch)
        self.timeout, self.retries = timeout, retries
        self.integration_guards = dict(integration_guards or {})
        if any(float(value) < 0 for value in self.integration_guards.values()):
            raise ValueError("Integration guards cannot be negative")
        self.source_records, self._record_keys = [], set()
        self._memory, self._window_cache = OrderedDict(), OrderedDict()
        self._offline_all, self._series_info = None, {}

    def metadata(self, feature):
        if feature not in FEATURES:
            raise KeyError(feature)
        return {"feature": feature,
                "quantity": "total line-of-sight unsigned photospheric magnetic flux" if feature == "USFLUXL" else "native logarithmic Schrijver R index near polarity inversion lines",
                "units": "Mx" if feature == "USFLUXL" else "native log10 R index; archive keyword labels Maxwell; no exponentiation applied",
                "source": "native JSOC SMARP/SHARP keyword records, not reconstructed SMHARP",
                "native_series": [SMARP.series, SHARP.series],
                "native_cadence_minutes": {"SMARP": 96, "SHARP": 12},
                "era_switch_utc": SHARP_START.isoformat(),
                "calibration": "Native MDI and HMI values; no regression, scaling, or assertion of cross-instrument equivalence",
                "time_policy": "TAI converted to UTC; max(T_REC,T_OBS) plus integration guard; backward hold only",
                "integration_guard_minutes": {spec.era: self.integration_guards.get(spec.era, spec.integration_guard_minutes) for spec in (SMARP, SHARP)},
                "publication_latency": "Definitive retrospective products; real-time publication availability is not reconstructed",
                "max_observation_age_cadences": self.max_age_cadences,
                "region_policy": "exact NOAA identifier membership; multiple-region patches excluded" if not self.allow_multi_region_patch else "exact NOAA identifier membership; multiple-region native patches explicitly flagged",
                "quality_policy": {"SMARP": "0 <= QUALITY < 262144", "SHARP": "0 <= QUALITY < 65536"},
                "quality_reference": QUALITY_REFERENCE, "parameter_reference": PARAMETER_REFERENCE,
                "point_status": {"O": "newly available native observation in this five-minute bin", "H": "backward held native value", "G": "missing, stale or invalid parameter", "Q": "native quality rejected", "A": "ambiguous patch association", "M": "missing NOAA active-region identifier", "S": "outside native coverage"},
                "offline_columns": ["series", "T_REC", "T_OBS", "TARPNUM or HARPNUM", "NOAA_AR", "NOAA_ARS", "QUALITY", *FEATURES]}

    def _record(self, value):
        key = json.dumps(value, sort_keys=True, default=str)
        if key not in self._record_keys:
            self.source_records.append(value)
            self._record_keys.add(key)

    def _request(self, params):
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / "raw" / f"{key}.json"
        checksum = path.with_suffix(".sha256")
        body = None
        if path.exists() and checksum.exists():
            candidate = path.read_bytes()
            if hashlib.sha256(candidate).hexdigest() == checksum.read_text().strip():
                body = candidate
        error = None
        if body is None:
            for attempt in range(self.retries):
                try:
                    response = requests.get(JSOC_URL, params=params, timeout=(20, self.timeout))
                    response.raise_for_status()
                    candidate = response.content
                    payload = json.loads(candidate)
                    if not isinstance(payload, dict):
                        raise MagneticSourceError(f"JSOC error: {str(payload)[:250]}")
                    if payload.get("status") != 0:
                        raise MagneticQueryError(payload.get("status"), payload.get("error", ""))
                    body = candidate
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix(".part")
                    temp.write_bytes(body)
                    temp.replace(path)
                    checksum.write_text(hashlib.sha256(body).hexdigest())
                    break
                except MagneticQueryError:
                    raise
                except (requests.RequestException, ValueError, MagneticSourceError) as exc:
                    error = exc
                    if attempt + 1 < self.retries:
                        time.sleep(min(2 ** attempt, 5))
            if body is None:
                raise MagneticSourceError(f"Native JSOC keywords unavailable: {error}. Retry or supply --magnetic-csv; no measurements have been fabricated.")
        payload = json.loads(body)
        self._record({"url": JSOC_URL, "query": params, "sha256": hashlib.sha256(body).hexdigest(), "native_keywords_only": True})
        return payload

    def _fetch(self, spec, start, stop):
        if spec.series not in self._series_info:
            info = self._request({"op": "series_struct", "ds": spec.series})
            keywords = {item["name"]: item for item in info.get("keywords", [])}
            required = {"T_REC", "T_OBS", spec.patch_key, "NOAA_AR", "NOAA_ARS", "QUALITY", *FEATURES}
            if not required.issubset(keywords):
                raise MagneticSourceError(f"JSOC series metadata lacks {sorted(required-set(keywords))}")
            self._series_info[spec.series] = keywords
            self._record({"series": spec.series, "keyword_metadata": {name: keywords[name] for name in FEATURES}, "quality_limit": spec.quality_limit})
        def tai(stamp):
            return Time(stamp.to_pydatetime(), scale="utc").tai.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        def query(low, high):
            if self.progress:
                self.progress(f"[JSOC QUERY] {spec.era} {low.isoformat()} through {high.isoformat()}")
            params = {"op": "rs_list", "ds": f"{spec.series}[][{tai(low)}-{tai(high)}]",
                      "key": ",".join(["T_REC", "T_OBS", spec.patch_key, "NOAA_AR", "NOAA_ARS", "QUALITY", *FEATURES]),
                      "seg": "**NONE**", "link": "**NONE**"}
            try:
                return parse_keyword_payload(self._request(params), spec)
            except MagneticQueryError as exc:
                # JSOC returns status 6 for some large all-patch requests.
                # Split them, but never reinterpret a failed daily query as a gap.
                if exc.status != 6 or high - low <= pd.Timedelta(days=1):
                    raise
                middle = (low + (high-low)/2).floor("s")
                return pd.concat([query(low, middle), query(middle, high)], ignore_index=True)
        chunks = []
        cursor = start
        while cursor < stop:
            high = min(cursor + pd.Timedelta(days=7), stop)
            chunks.append(query(cursor, high))
            cursor = high
        # JSOC endpoint ranges include both boundaries. Retain one primary-key
        # record at split points; month filtering applies exact UTC bounds later.
        return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["T_REC", spec.patch_key]).reset_index(drop=True)

    def _month(self, spec, month):
        key = (spec.series, month.strftime("%Y-%m"))
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        stop = month + pd.offsets.MonthBegin(1)
        if self.progress:
            self.progress(f"[MAGNETIC] {spec.era} {key[1]}: native USFLUXL/R_VALUE keywords")
        if self.native_csv is not None:
            path = self.native_csv
            if path.is_dir():
                path = path / f"{spec.era}_{key[1]}.csv"
                if not path.exists():
                    raise MagneticSourceError(f"Offline native keyword month missing: {path}")
                raw = pd.read_csv(path, keep_default_na=False)
            else:
                if self._offline_all is None:
                    self._offline_all = pd.read_csv(path, keep_default_na=False)
                    if "series" not in self._offline_all:
                        raise MagneticSourceError("A single native keyword CSV requires a series column")
                raw = self._offline_all[self._offline_all.series.eq(spec.series)]
            self._record({"source": "offline_native_keyword_csv", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "series": spec.series})
        else:
            raw = self._fetch(spec, month, stop)
        data = normalize_native(raw, spec)
        data = data[(data.record_time >= month) & (data.record_time < stop)].copy()
        if spec.era in self.integration_guards:
            data["available_at"] += pd.Timedelta(minutes=self.integration_guards[spec.era] - spec.integration_guard_minutes)
        index = {}
        for position, members in enumerate(data.members):
            for member in members:
                index.setdefault(member, []).append(position)
        grouped = {}
        for member, positions in index.items():
            block = data.iloc[positions].drop_duplicates(subset=["series", "patch", "T_REC", "T_OBS", "USFLUXL", "R_VALUE", "QUALITY", "NOAA_AR", "NOAA_ARS"]).copy()
            multiple = block.available_at.duplicated(keep=False)
            block["association_ok"] = ~multiple & block.members.map(lambda value: self.allow_multi_region_patch or len(value) == 1)
            # Collapse once per cached month/region, not for every flare window.
            # Conflicting simultaneous patches remain explicitly ambiguous.
            grouped[member] = block.drop_duplicates("available_at", keep="first").sort_values("available_at").reset_index(drop=True)
        self._record({"series": spec.series, "month": key[1], "native_records": len(data), "quality_rejected_records": int((~data.quality_ok).sum()), "quality_flagged_records": int(data.native_quality.fillna(0).ne(0).sum()), "reconstructed": False})
        self._memory[key] = grouped
        while len(self._memory) > self.memory_months:
            self._memory.popitem(last=False)
        return grouped

    def _select(self, spec, grid, noaa_ar):
        low = grid.min() - pd.Timedelta(minutes=spec.cadence_minutes * self.max_age_cadences + spec.integration_guard_minutes)
        current = low.normalize().replace(day=1)
        frames = []
        while current <= grid.max():
            frame = self._month(spec, current).get(noaa_ar)
            if frame is not None and not frame.empty:
                times = frame.available_at.astype("int64").to_numpy()
                first = max(0, int(np.searchsorted(times, low.value, side="left")) - 1)
                last = int(np.searchsorted(times, grid.max().value, side="right"))
                frames.append(frame.iloc[first:last])
            current += pd.offsets.MonthBegin(1)
        empty = pd.DataFrame(index=grid)
        if not frames:
            return empty
        data = pd.concat(frames, ignore_index=True).sort_values("available_at")
        if data.empty:
            return empty
        data.loc[data.available_at.duplicated(keep=False), "association_ok"] = False
        data = data.drop_duplicates("available_at", keep="first")
        left = pd.DataFrame({"grid_time": grid})
        selected = pd.merge_asof(left, data, left_on="grid_time", right_on="available_at", direction="backward")
        return selected.set_index("grid_time")

    def window(self, feature, onset, noaa_ar):
        self.metadata(feature)
        onset = _utc(onset)
        end = onset.ceil("5min")
        grid = pd.date_range(end - pd.Timedelta(hours=24), end - pd.Timedelta(minutes=5), freq="5min")
        ar = _ar_number(noaa_ar)
        cache_key = (onset.isoformat(), ar)
        if cache_key in self._window_cache:
            selected = self._window_cache[cache_key]
        else:
            selected = pd.DataFrame(index=grid)
            for spec, mask in [(SMARP, grid < SHARP_START), (SHARP, grid >= SHARP_START)]:
                if mask.any() and ar is not None:
                    part = self._select(spec, grid[mask], ar)
                    part["association_noaa_ar"] = ar
                    # Some historical flare catalogs retain only the last four
                    # NOAA digits after the 10000 rollover. Expand only when
                    # an actual native record confirms that full identifier.
                    has_exact = "observation_time" in part and part.observation_time.notna().any()
                    if not has_exact and onset.year >= 2002 and ar < 10000:
                        expanded = self._select(spec, grid[mask], ar + 10000)
                        if "observation_time" in expanded and expanded.observation_time.notna().any():
                            part = expanded
                            part["association_noaa_ar"] = ar + 10000
                    selected = selected.combine_first(part)
            selected = selected.reindex(grid)
            self._window_cache[cache_key] = selected
            while len(self._window_cache) > 4:
                self._window_cache.popitem(last=False)
        q = pd.DataFrame(index=grid)
        q["point_status"], q["reason"] = "G", "no_matching_native_record"
        q["source_product"] = np.where(grid < SHARP_START, SMARP.series, SHARP.series)
        q["source_era"] = np.where(grid < SHARP_START, "SMARP", "SHARP")
        q["source_satellite"] = np.where(grid < SHARP_START, "SOHO/MDI", "SDO/HMI")
        q["native_cadence_minutes"] = np.where(grid < SHARP_START, 96, 12)
        q["requested_noaa_ar"] = ar
        q["association_noaa_ar"] = selected.get("association_noaa_ar", pd.Series(np.nan, index=grid))
        q["noaa_identifier_expanded"] = q.association_noaa_ar.notna() & q.association_noaa_ar.ne(ar)
        q["max_age_minutes"] = q.native_cadence_minutes * self.max_age_cadences
        q["native_observation_time"] = pd.to_datetime(selected.get("observation_time", pd.Series(pd.NaT, index=grid)), utc=True)
        q["native_record_time"] = pd.to_datetime(selected.get("record_time", pd.Series(pd.NaT, index=grid)), utc=True)
        q["available_at"] = pd.to_datetime(selected.get("available_at", pd.Series(pd.NaT, index=grid)), utc=True)
        q["age_minutes"] = (pd.Series(grid, index=grid) - q.native_observation_time).dt.total_seconds() / 60
        q["native_quality"] = selected.get("native_quality", pd.Series(np.nan, index=grid))
        q["quality_nonzero"] = q.native_quality.notna() & q.native_quality.ne(0)
        q["matched_patch"] = selected.get("patch", pd.Series("", index=grid))
        q["matched_noaa_ars"] = selected.get("NOAA_ARS", pd.Series("", index=grid))
        q["region_count"] = selected.get("members", pd.Series(None, index=grid, dtype=object)).map(lambda value: len(value) if isinstance(value, tuple) else 0)
        matched = q.native_observation_time.notna()
        fresh = matched & q.age_minutes.ge(0) & q.age_minutes.le(q.max_age_minutes) & q.available_at.le(pd.Series(grid, index=grid))
        association_ok = selected.get("association_ok", pd.Series(False, index=grid)).eq(True)
        quality_ok = selected.get("quality_ok", pd.Series(False, index=grid)).eq(True)
        value = pd.to_numeric(selected.get(feature, pd.Series(np.nan, index=grid)), errors="coerce")
        valid = fresh & association_ok & quality_ok & value.notna()
        recent = (pd.Series(grid, index=grid) - q.available_at).dt.total_seconds().lt(300)
        q.loc[valid & recent, ["point_status", "reason"]] = ["O", "new_native_observation"]
        q.loc[valid & ~recent, ["point_status", "reason"]] = ["H", "causal_backward_hold"]
        q.loc[matched & ~fresh, "reason"] = "native_observation_too_old"
        q.loc[fresh & ~association_ok, ["point_status", "reason"]] = ["A", "ambiguous_or_multi_region_patch"]
        q.loc[fresh & association_ok & ~quality_ok, ["point_status", "reason"]] = ["Q", "native_quality_rejected"]
        q.loc[fresh & association_ok & quality_ok & value.isna(), "reason"] = "native_parameter_missing"
        if ar is None:
            q[["point_status", "reason"]] = ["M", "missing_noaa_active_region"]
        q.loc[grid < SMARP_START, ["point_status", "reason"]] = ["S", "outside_native_coverage"]
        values = value.where(valid)
        values.name = feature
        values.attrs.update(self.metadata(feature))
        return values, q


def main(argv=None):
    """First-party native-keyword export for transport to an offline server."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", choices=list(SPECS), required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="Exclusive UTC end")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="data/cache/magnetic")
    args = parser.parse_args(argv)
    store = MagneticStore(args.cache_dir)
    spec = SPECS[args.series]
    frame = store._fetch(spec, _utc(args.start), _utc(args.end))
    frame["series"] = spec.series
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"[DONE] {len(frame)} native keyword records: {output}")


if __name__ == "__main__":
    main()

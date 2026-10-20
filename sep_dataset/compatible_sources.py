"""Measured sources for the ten original MEMPSEP predictors.

The legacy GOES EPS/EPEAD P4/P5/P6 differential channels are deliberately
unavailable after GOES-15 ceased operating. SGPS channels and integral proton
thresholds are different quantities and are never substituted for them.
"""
from __future__ import annotations

from collections import OrderedDict
from io import StringIO
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np
import pandas as pd
import requests


FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6")
OMNI_FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F")
PROTON_FEATURES = ("P4", "P5", "P6")
INTEGRAL_COLUMNS = ("GT5", "GT10", "GT30", "GT60", "GT100")
LEGACY_END = pd.Timestamp("2020-03-05T00:00:00Z")
NOAA_ARCHIVE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg"
ISWA = "https://iswa.ccmc.gsfc.nasa.gov/hapi"
CDAWEB = "https://cdaweb.gsfc.nasa.gov/hapi"
CHANNEL_DOCUMENTATION = "https://www.ngdc.noaa.gov/stp/satellite/goes/datanotes.html"
XRS_DOCUMENTATION = "https://www.ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf"


class SourceDataError(RuntimeError):
    """An expected source could not be downloaded or validated."""


def _hapi_no_data(body: bytes, url: str, http_status: int | None = None) -> dict | None:
    """Recognize genuine empty intervals, never a generic 404 or other failure.

    HAPI permits an empty HTTP-200 CSV or a JSON 1200/1201 empty data array.
    ISWA additionally uses HTTP 404/code 1406 with the exact no-data message
    below, although standard HAPI reserves 1406 for an unknown dataset. Only
    that specific ISWA response is accepted; unknown IDs and HTTP errors fail.
    The caller validates the dataset metadata before requesting its data.
    """
    if not body.strip():
        return {"code": 1201, "message": "Empty HAPI CSV interval"} if http_status in (None, 200) else None
    try:
        payload = json.loads(body.decode("utf-8-sig"))
    except (ValueError, UnicodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("HAPI"), str) or payload.get("data") != []:
        return None
    status = payload.get("status")
    if not isinstance(status, dict):
        return None
    if http_status in (None, 200) and status.get("code") in (1200, 1201):
        return status
    if (url == ISWA + "/data" and http_status in (None, 404)
            and status.get("code") == 1406
            and str(status.get("message", "")).strip() == "No data available for the specified ID and time range"):
        return status
    return None


def _utc(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize("UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")


def _month(value: object) -> pd.Timestamp:
    stamp = _utc(value)
    return pd.Timestamp(year=stamp.year, month=stamp.month, day=1, tz="UTC")


def _months(start: pd.Timestamp, stop: pd.Timestamp):
    current = _month(start)
    while current < stop:
        yield current
        current += pd.offsets.MonthBegin(1)


def _grid(start: pd.Timestamp, stop: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, stop, freq="5min", inclusive="left", name="Time")


def parse_noaa_csv(text: str) -> pd.DataFrame:
    """Read the data section, ignoring NOAA's metadata and orbit tables."""
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith("time_tag,")]
    if len(headers) != 1:
        raise SourceDataError("NOAA CSV lacks a unique time_tag data header")
    frame = pd.read_csv(StringIO("\n".join(lines[headers[0]:])), skipinitialspace=True)
    frame["time_tag"] = pd.to_datetime(frame["time_tag"], utc=True, errors="coerce")
    if frame.empty or frame["time_tag"].isna().any():
        raise SourceDataError("NOAA CSV is empty or contains invalid timestamps")
    frame = frame.set_index("time_tag").sort_index()
    if frame.index.has_duplicates:
        raise SourceDataError("NOAA CSV contains duplicate timestamps")
    frame.index.name = "Time"
    return frame


def _numeric(series: pd.Series, fill: object = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if fill not in (None, "null", "None"):
        try:
            values = values.mask(np.isclose(values, float(fill), rtol=1e-7, atol=1e-12))
        except (TypeError, ValueError):
            pass
    return values.mask(~np.isfinite(values) | (values <= -9999) | (values.abs() >= 1e29))


def parse_legacy_goes(text: str, kind: str, satellite: int = 15) -> pd.DataFrame:
    """Use raw differential flux or corrected integral flux, with NOAA QC."""
    raw = parse_noaa_csv(text)
    result = pd.DataFrame(index=raw.index)
    channels = PROTON_FEATURES if kind == "protons" else INTEGRAL_COLUMNS
    for feature in channels:
        stem = f"{feature}W" if kind == "protons" else f"ZP{feature}W"
        column = f"{stem}_UNCOR_FLUX" if kind == "protons" else stem
        quality = f"{stem}_QUAL_FLAG"
        if column not in raw or quality not in raw:
            raise SourceDataError(f"NOAA GOES-{satellite} CSV missing {column} or {quality}")
        values = _numeric(raw[column], -99999)
        values = values.mask(pd.to_numeric(raw[quality], errors="coerce") != 0)
        count = f"{stem}_NUM_PTS"
        if count in raw:
            values = values.mask(pd.to_numeric(raw[count], errors="coerce") <= 0)
        result[feature] = values.mask(values < 0)
    result["source_satellite"] = f"GOES-{satellite}"
    result["source_product"] = "NOAA_EPEAD_P17EW_UNCOR_W" if kind == "protons" else "NOAA_EPEAD_CPFLUX_ZPGT_W"
    result["channel_status"] = "measured_legacy_channel" if kind == "protons" else "corrected_integral_measurement"
    return result


class CompatibleSourceStore:
    """Validated monthly cache; strict original bands and explicit provenance.

    ``range`` returns native measurements, without any interpolation or fill.
    Structural absence is NaN with ``outside_original_channel_coverage`` in
    ``quality_range``. Network or parse errors raise instead of becoming gaps.
    The builder decides how to fill limited observational gaps and must retain
    structural gaps. Native timestamp semantics are retained and documented;
    the builder must enforce its pre-flare availability rule.
    """

    def __init__(self, cache_dir: str | Path = "data/cache/compatible_sources", *,
                 channel_policy: str = "strict", timeout: int = 120, retries: int = 3,
                 progress: Callable[[str], None] | None = print, memory_months: int = 6):
        if channel_policy != "strict":
            raise ValueError("Only strict original channels are supported; SGPS is not a P4/P5/P6 substitute")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.channel_policy = channel_policy
        self.timeout = timeout
        self.retries = retries
        self.progress = progress
        self.memory_months = max(1, memory_months)
        self._memory: OrderedDict[tuple[str, str], pd.DataFrame] = OrderedDict()
        self.source_records: list[dict] = []
        self._record_keys: set[str] = set()
        self._captured_records: list[dict] | None = None

    def metadata(self, feature: str) -> dict:
        if feature not in FEATURES:
            raise KeyError(feature)
        units = {"Vx": "km/s", "flow_speed": "km/s", "proton_density": "cm^-3", "T": "K", "F": "nT", "Xs": "W/m^2", "Xl": "W/m^2"}
        if feature in PROTON_FEATURES:
            bands = {"P4": [15, 40], "P5": [40, 80], "P6": [80, 165]}
            return {"feature": feature, "units": "protons/(cm^2 s sr MeV)",
                    "quantity": "uncorrected differential proton flux", "GOES8_12_reference_nominal_band_MeV": bands[feature],
                    "GOES15_native_reported_energy_MeV": {"P4": 30.6, "P5": 63.1, "P6": 165.0}[feature],
                    "source": "NOAA GOES-15 EPEAD P17EW W-labelled detector", "instrument_response": "Native GOES-15 EPEAD channel; differs from older EPS energy response/calibration. The old nominal reference band is not an ideal rectangular response or an assertion of cross-satellite equivalence.",
                    "coverage_end_exclusive": LEGACY_END.isoformat(), "after_coverage": "outside_original_channel_coverage; NaN; no SGPS substitution",
                    "extension_eras": [{"start": "2014-01-01", "stop_exclusive": LEGACY_END.isoformat(), "product": "GOES-15 EPEAD P17EW uncorrected W-labelled detector", "status": "native_legacy_measurement_subject_to_QC"}, {"start": LEGACY_END.isoformat(), "stop_exclusive": None, "product": None, "status": "outside_original_channel_coverage"}],
                    "detector_note": "W-labelled A detector can face east or west depending on satellite yaw; not assumed to be primary or fixed west-facing",
                    "documentation": CHANNEL_DOCUMENTATION, "channel_policy": "strict", "bin_lookahead_minutes": 5}
        if feature in ("Xs", "Xl"):
            return {"feature": feature, "units": units[feature], "wavelength_nm": [0.05, 0.4] if feature == "Xs" else [0.1, 0.8],
                    "source": "NASA ISWA GOES primary operational XRS", "native_cadence": "1min", "output_cadence": "5min trailing arithmetic mean of available valid minutes, right-labelled: (t-5min, t]", "bin_lookahead_minutes": 0,
                    "calibration": "Native archived operational values. GOES-8–15 SWPC scaling and GOES-R calibration differ; no assumed rescaling applied.",
                    "documentation": XRS_DOCUMENTATION}
        return {"feature": feature, "units": units[feature], "source": "NASA CDAWeb OMNI_HRO2_5MIN", "parameter": feature, "cadence": "5min", "timing": "OMNI propagated to bow-shock nose; native reported timestamps retained", "bin_lookahead_minutes": 5}

    info = metadata

    def range(self, feature: str, start: object, stop: object) -> pd.Series:
        self.metadata(feature)
        group = "omni" if feature in OMNI_FEATURES else "xrs" if feature in ("Xs", "Xl") else "protons"
        frame = self._range(group, _utc(start), _utc(stop))
        result = frame[feature].copy()
        result.name = feature
        result.attrs.update(self.metadata(feature))
        result.attrs["structural_missing"] = frame["channel_status"].eq("outside_original_channel_coverage").to_numpy()
        structural = result.attrs["structural_missing"]
        result.attrs["source_status"] = "outside_original_channel_coverage" if structural.all() else "mixed_coverage" if structural.any() else "measured_with_possible_observational_gaps"
        result.attrs["source_records"] = list(self.source_records)
        return result

    def quality_range(self, feature: str, start: object, stop: object) -> pd.DataFrame:
        self.metadata(feature)
        group = "omni" if feature in OMNI_FEATURES else "xrs" if feature in ("Xs", "Xl") else "protons"
        frame = self._range(group, _utc(start), _utc(stop))
        result = frame[["channel_status", "source_product", "source_satellite"]].copy()
        result["measurement_missing"] = frame[feature].isna()
        return result

    def integral_range(self, start: object, stop: object) -> pd.DataFrame:
        frame = self._range("integral", _utc(start), _utc(stop))
        result = frame[list(INTEGRAL_COLUMNS)].copy()
        result.attrs["units"] = "protons/(cm^2 s sr) [pfu]"
        result.attrs["source_records"] = list(self.source_records)
        result.attrs["provenance"] = frame[["source_product", "source_satellite", "channel_status"]].drop_duplicates().to_dict("records")
        return result

    def _range(self, group: str, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
        if stop <= start:
            raise ValueError("stop must be later than start")
        frames = [self._load_month(group, month) for month in _months(start, stop)]
        frame = pd.concat(frames).sort_index()
        if frame.index.has_duplicates:
            raise SourceDataError(f"Duplicate timestamps in combined {group} source")
        return frame[(frame.index >= start) & (frame.index < stop)]

    def _record(self, record: dict):
        key = json.dumps(record, sort_keys=True)
        if self._captured_records is not None and record not in self._captured_records:
            self._captured_records.append(record)
        if key not in self._record_keys:
            self.source_records.append(record)
            self._record_keys.add(key)

    def _load_month(self, group: str, start: pd.Timestamp) -> pd.DataFrame:
        key = (group, start.strftime("%Y-%m"))
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        stop = start + pd.offsets.MonthBegin(1)
        parsed = self.cache_dir / "parsed" / f"{group}_{key[1]}_strict_v3.csv"
        manifest = parsed.with_suffix(".json")
        cached = False
        if parsed.exists() and manifest.exists():
            try:
                saved = json.loads(manifest.read_text())
                if saved.get("parsed_sha256") != hashlib.sha256(parsed.read_bytes()).hexdigest():
                    raise SourceDataError("Parsed cache checksum is absent or does not match")
                frame = pd.read_csv(parsed, index_col="Time", parse_dates=["Time"])
                frame.index = pd.to_datetime(frame.index, utc=True)
                if not frame.index.equals(_grid(start, stop)):
                    raise SourceDataError("Parsed cache has an incomplete or invalid time grid")
                for record in saved["sources"]:
                    self._record(record)
                cached = True
            except (OSError, ValueError, KeyError, TypeError, SourceDataError) as exc:
                if self.progress:
                    self.progress(f"[CACHE REFRESH] {group} {key[1]}: {exc}")
        if not cached:
            if self.progress:
                self.progress(f"[SOURCE] {group} {key[1]}")
            self._captured_records = []
            if group == "omni":
                frame = self._hapi(CDAWEB, "OMNI_HRO2_5MIN", list(OMNI_FEATURES), start, stop)
                frame["source_product"] = "OMNI_HRO2_5MIN"
                frame["source_satellite"] = "OMNI multi-spacecraft"
                frame["channel_status"] = "measured_omni"
            elif group == "xrs":
                frame = self._hapi(ISWA, "goesp_xray_flux_P1M", ["Short_Wave", "Long_Wave", "satellite"], start - pd.Timedelta(minutes=5), stop)
                satellite = frame["satellite"].fillna("unknown").resample("5min", closed="right", label="right").first()
                frame = frame[["Short_Wave", "Long_Wave"]].resample("5min", closed="right", label="right").mean().rename(columns={"Short_Wave": "Xs", "Long_Wave": "Xl"})
                frame["source_product"] = "ISWA_GOES_PRIMARY_XRS_NATIVE_OPERATIONAL"
                frame["source_satellite"] = satellite
                frame["channel_status"] = "native_operational_calibration_era_not_homogenized"
            elif group in ("protons", "integral"):
                if start >= LEGACY_END and group == "protons":
                    frame = pd.DataFrame(np.nan, index=_grid(start, stop), columns=PROTON_FEATURES)
                    frame["channel_status"] = "outside_original_channel_coverage"
                    frame["source_product"] = "no_exact_legacy_EPS_EPEAD_channels"
                    frame["source_satellite"] = "unavailable"
                    self._record({"source": "structural_unavailability", "month": key[1], "features": list(PROTON_FEATURES), "reason": "GOES-15 ended March 2020; SGPS has different differential energy responses"})
                elif start >= LEGACY_END:
                    frame = self._modern_integral(start, stop)
                else:
                    frame = self._legacy(group, start, stop)
                    if stop > LEGACY_END:
                        if group == "integral":
                            modern = self._modern_integral(LEGACY_END, stop)
                            frame = pd.concat([frame[frame.index < LEGACY_END], modern])
                        else:
                            outside = frame.index >= LEGACY_END
                            frame.loc[outside, list(PROTON_FEATURES)] = np.nan
                            frame.loc[outside, "channel_status"] = "outside_original_channel_coverage"
                            frame.loc[outside, "source_satellite"] = "unavailable"
                            frame.loc[outside, "source_product"] = "no_exact_legacy_EPS_EPEAD_channels"
            else:
                raise KeyError(group)
            frame = frame.reindex(_grid(start, stop))
            if frame.index.has_duplicates:
                raise SourceDataError("Source has duplicate five-minute timestamps")
            for name in ("channel_status", "source_product", "source_satellite"):
                frame[name] = frame[name].fillna("missing_observation" if name == "channel_status" else "unknown")
            # The final NOAA monthly CSV is truncated on March 4, although its
            # filename ends March 31. Apply coverage after reindexing so those
            # absent rows cannot become ordinary gaps eligible for filling.
            if group == "protons":
                outside = frame.index >= LEGACY_END
                frame.loc[outside, list(PROTON_FEATURES)] = np.nan
                frame.loc[outside, "channel_status"] = "outside_original_channel_coverage"
                frame.loc[outside, "source_satellite"] = "unavailable"
                frame.loc[outside, "source_product"] = "no_exact_legacy_EPS_EPEAD_channels"
            parsed.parent.mkdir(parents=True, exist_ok=True)
            part = parsed.with_suffix(".csv.part")
            frame.to_csv(part, index_label="Time")
            part.replace(parsed)
            manifest_part = manifest.with_suffix(".json.part")
            manifest_part.write_text(json.dumps({"sources": self._captured_records, "parsed_sha256": hashlib.sha256(parsed.read_bytes()).hexdigest()}, indent=2))
            manifest_part.replace(manifest)
            self._captured_records = None
        self._memory[key] = frame
        while len(self._memory) > self.memory_months:
            self._memory.popitem(last=False)
        return frame

    def _modern_integral(self, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
        fields = ["P5", "P10", "P30", "P60", "P100", "satelliteProton"]
        frame = self._hapi(ISWA, "goesp_part_flux_P5M", fields, start, stop)
        frame = frame.rename(columns={f"P{v}": f"GT{v}" for v in (5, 10, 30, 60, 100)})
        frame["source_product"] = "ISWA_GOES_PRIMARY_OPERATIONAL_INTEGRAL"
        frame["source_satellite"] = frame.pop("satelliteProton").fillna("unknown")
        frame["channel_status"] = "native_operational_integral"
        return frame

    def _legacy(self, group: str, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
        product = "p17ew" if group == "protons" else "cpflux"
        # Fixed, explicit detector policy; never call this the primary satellite.
        satellite = 15
        last = (stop - pd.Timedelta(days=1)).strftime("%Y%m%d")
        name = f"g{satellite}_epead_{product}_5m_{start:%Y%m%d}_{last}.csv"
        url = f"{NOAA_ARCHIVE}/{start:%Y/%m}/goes{satellite}/csv/{name}"
        body = self._download(url, kind="legacy_csv")
        frame = parse_legacy_goes(body.decode("utf-8-sig"), group, satellite)
        self._record({"url": url, "sha256": hashlib.sha256(body).hexdigest(), "product": product, "satellite": f"GOES-{satellite}", "detector": "W-labelled A (yaw-dependent look direction)", "uncorrected_differential": group == "protons", "corrected_integral": group == "integral", "units": "protons/(cm^2 s sr MeV)" if group == "protons" else "pfu"})
        return frame

    def _hapi(self, server: str, dataset: str, wanted: list[str], start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
        # ISWA rejects requests longer than 31 days, including a month plus
        # the few preceding minutes needed for a trailing XRS boundary bin.
        maximum = pd.Timedelta(days=31)
        if stop - start > maximum:
            frames = []
            cursor = start
            while cursor < stop:
                end = min(cursor + maximum, stop)
                frames.append(self._hapi(server, dataset, wanted, cursor, end))
                cursor = end
            return pd.concat(frames).sort_index()
        info_body = self._download(server + "/info", {"id": dataset}, kind="json")
        info = json.loads(info_body)
        metadata = {p["name"]: p for p in info.get("parameters", [])}
        missing = set(wanted) - set(metadata)
        if missing:
            raise SourceDataError(f"HAPI {dataset} missing configured parameters: {sorted(missing)}")
        # HAPI requires the subset in server metadata order.
        selected = [p["name"] for p in info["parameters"] if p["name"] in wanted]
        params = {"id": dataset, "parameters": ",".join(selected), "time.min": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "time.max": stop.strftime("%Y-%m-%dT%H:%M:%SZ"), "format": "csv"}
        body = self._download(server + "/data", params, kind="hapi_csv")
        gap = _hapi_no_data(body, server + "/data")
        provenance = {"url": server + "/data", "query": params, "sha256": hashlib.sha256(body).hexdigest(), "dataset": dataset, "metadata_coverage_start": info.get("startDate"), "metadata_coverage_stop": info.get("stopDate"), "parameters": {k: {a: metadata[k].get(a) for a in ("units", "fill", "description")} for k in selected}}
        if gap is not None:
            index = pd.DatetimeIndex([], tz="UTC", name="Time")
            frame = pd.DataFrame({name: pd.Series(index=index, dtype="object" if metadata[name].get("type") == "string" else "float64") for name in selected}, index=index)
            self._record({**provenance, "source_status": "no_data_available", "hapi_status": gap})
            if self.progress:
                self.progress(f"[SOURCE GAP] {dataset} {params['time.min']} through {params['time.max']}: {gap.get('message', 'No data available')}")
            return frame
        frame = pd.read_csv(StringIO(body.decode("utf-8-sig")), header=None)
        if frame.empty or frame.shape[1] != len(selected) + 1:
            raise SourceDataError(f"HAPI {dataset} invalid column count or empty data")
        frame.columns = ["Time"] + selected
        frame["Time"] = pd.to_datetime(frame["Time"], utc=True, errors="coerce")
        if frame["Time"].isna().any() or frame["Time"].duplicated().any():
            raise SourceDataError(f"HAPI {dataset} invalid or duplicate timestamps")
        frame = frame.set_index("Time").sort_index()
        frame = frame[(frame.index >= start) & (frame.index < stop)]
        for name in selected:
            if metadata[name].get("type") != "string":
                frame[name] = _numeric(frame[name], metadata[name].get("fill"))
                if dataset != "OMNI_HRO2_5MIN":
                    frame[name] = frame[name].mask(frame[name] < 0)
        self._record(provenance)
        return frame

    def _download(self, url: str, params: dict | None = None, *, kind: str) -> bytes:
        identity = json.dumps([url, params], sort_keys=True)
        path = self.cache_dir / "raw" / hashlib.sha256(identity.encode()).hexdigest()
        if path.exists():
            return path.read_bytes()
        error = None
        for attempt in range(self.retries):
            try:
                response = requests.get(url, params=params, timeout=(20, self.timeout))
                body = response.content
                gap = _hapi_no_data(body, url, getattr(response, "status_code", 200)) if kind == "hapi_csv" else None
                if gap is None:
                    try:
                        response.raise_for_status()
                    except requests.HTTPError as exc:
                        raise SourceDataError(f"{exc}: {response.text[:240]}") from exc
                text = body.decode("utf-8-sig")
                if kind == "json":
                    payload = json.loads(text)
                    if "parameters" not in payload:
                        raise SourceDataError(f"Invalid HAPI metadata: {text[:200]}")
                elif kind == "hapi_csv":
                    if gap is None and not re.match(r"^\s*\d{4}-\d{2}-\d{2}T", text):
                        raise SourceDataError(f"Invalid HAPI CSV: {text[:200]}")
                elif "time_tag," not in text or "<html" in text[:500].lower():
                    raise SourceDataError("Invalid NOAA legacy CSV response")
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".part")
                temp.write_bytes(body)
                temp.replace(path)
                return body
            except (requests.RequestException, ValueError, SourceDataError) as exc:
                error = exc
                if self.progress:
                    self.progress(f"[SOURCE RETRY {attempt + 1}/{self.retries}] {url}: {exc}")
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 5))
        raise SourceDataError(f"Source unavailable after {self.retries} attempts: {url}: {error}")

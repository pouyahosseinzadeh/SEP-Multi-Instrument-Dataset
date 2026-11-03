"""Explicit spectral continuation of legacy proton predictors using NOAA SGPS.

These are differential point-flux estimates at 30.6, 63.1 and 165 MeV. They
are NOT the native SGPS channels named P4/P5/P6, not integral fluxes, and not
demonstrated to reproduce an EPS/EPEAD detector response or calibration.
No time interpolation, extrapolation, zero replacement or median fill occurs.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin

import netCDF4
import numpy as np
import pandas as pd
import requests


TARGET_ENERGIES = {"P4": 30.6, "P5": 63.1, "P6": 165.0}
METHOD = "source_sgps_energy_interpolated"
UNITS = "protons/(cm^2 s sr MeV)"
ARCHIVE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
SPECIAL_2017 = "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/goesr/solar_proton_events/sgps_sep2017_event_data/se_sgps-l2-avg5m_g16_s20172440000000_e20172732355000_v2_0_0.nc"
DOCUMENTATION = "https://www.ncei.noaa.gov/products/goes-r-space-environment-in-situ"
CACHE_VERSION = "sgps-energy-v1"


class SGPSSourceError(RuntimeError):
    """Unexpected network, format, unit or quality-metadata failure."""


def _utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _grid(day):
    return pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left", name="Time")


def _array(variable):
    return np.asarray(np.ma.filled(np.ma.asarray(variable[:], dtype=float), np.nan), dtype=float)


def _atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_bytes(content)
    part.replace(path)


def _json(path, payload):
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _timestamps(variable, values=None):
    values = _array(variable) if values is None else values
    units = str(getattr(variable, "units", ""))
    match = re.fullmatch(r"(seconds|milliseconds|microseconds) since (.+)", units.strip())
    if not match:
        raise SGPSSourceError(f"Unsupported SGPS time units: {units}")
    unit = {"seconds": "s", "milliseconds": "ms", "microseconds": "us"}[match[1]]
    origin = _utc(match[2].replace("UTC", "").strip()).tz_localize(None)
    return pd.DatetimeIndex(pd.to_datetime(values, unit=unit, origin=origin, utc=True, errors="coerce"), name="Time")


def _flux_factor(variable):
    units = str(getattr(variable, "units", "")).lower().replace(" ", "")
    if "cm" not in units or "sr" not in units:
        raise SGPSSourceError(f"Unexpected differential flux units: {getattr(variable, 'units', '')}")
    if "kev" in units:
        return 1000.0
    if "mev" in units:
        return 1.0
    raise SGPSSourceError(f"No differential-energy unit in {getattr(variable, 'units', '')}")


def _energies(variable):
    units = str(getattr(variable, "units", "")).lower()
    if units not in ("kev", "mev"):
        raise SGPSSourceError(f"Unexpected energy units: {units}")
    return _array(variable) / (1000.0 if units == "kev" else 1.0)


def interpolate_energy(energy, flux, target, uncertainty=None):
    """Log-log interpolation between adjacent physical bands, without extrapolation.

    Zero, negative, missing or QC-rejected brackets remain missing. In
    particular, an invalid adjacent channel is not skipped to find a more
    convenient positive channel. The returned uncertainty propagates the
    supplied instrument errors assuming independent brackets; it omits spectral
    curvature, shared calibration errors and inter-instrument response errors.
    """
    energy, flux = np.asarray(energy, float), np.asarray(flux, float)
    if energy.ndim == 1:
        energy = np.broadcast_to(energy, flux.shape)
    if energy.shape != flux.shape or flux.ndim != 2:
        raise ValueError("Energy and flux must describe matching row-by-channel arrays")
    n = len(flux)
    result = {key: np.full(n, np.nan) for key in ("value", "uncertainty", "lower_energy", "upper_energy", "lower_flux", "upper_flux", "weight")}
    result["lower_index"], result["upper_index"] = np.full(n, -1, int), np.full(n, -1, int)
    error = np.asarray(uncertainty, float) if uncertainty is not None else None
    for row in range(n):
        e, f = energy[row], flux[row]
        if not np.isfinite(e).all() or not (np.diff(e) > 0).all() or target < e[0] or target > e[-1]:
            continue
        hi = int(np.searchsorted(e, target, side="left"))
        lo = hi if e[hi] == target else hi - 1
        if lo < 0 or not (np.isfinite(f[lo]) and np.isfinite(f[hi]) and f[lo] > 0 and f[hi] > 0):
            continue
        w = 0.0 if lo == hi else float(np.log(target / e[lo]) / np.log(e[hi] / e[lo]))
        value = float(np.exp((1 - w) * np.log(f[lo]) + w * np.log(f[hi])))
        result["value"][row], result["weight"][row] = value, w
        result["lower_index"][row], result["upper_index"][row] = lo, hi
        for key, val in (("lower_energy", e[lo]), ("upper_energy", e[hi]), ("lower_flux", f[lo]), ("upper_flux", f[hi])):
            result[key][row] = val
        if error is not None and np.isfinite(error[row, [lo, hi]]).all() and (error[row, [lo, hi]] >= 0).all():
            variance = (error[row, lo] / f[lo]) ** 2 if lo == hi else ((1 - w) * error[row, lo] / f[lo]) ** 2 + (w * error[row, hi] / f[hi]) ** 2
            result["uncertainty"][row] = value * np.sqrt(variance)
    return result


def _physical_west(dataset, n):
    """Select -X upright / +X inverted; reject ambiguous or transition yaw."""
    if "yaw_flip_flag" in dataset.variables:
        yaw = _array(dataset["yaw_flip_flag"])
        sensor = np.where(yaw == 0, 0, np.where(yaw == 2, 1, -1))
        basis = "native yaw 0=upright/-X west, 2=inverted/+X west; 1/unknown rejected"
    elif "YawFlipFlag" in dataset.variables:
        yaw = _array(dataset["YawFlipFlag"])
        # Older products expose an aggregate flag whose nonzero states cannot
        # unambiguously be translated into the newer 0/1/2 enum.
        sensor = np.where(yaw == 0, 0, -1)
        basis = "legacy aggregate yaw: zero proves upright/-X west; nonzero rejected as ambiguous"
    else:
        raise SGPSSourceError("SGPS source lacks yaw metadata needed to select a physical look direction")
    if len(sensor) != n:
        raise SGPSSourceError("Yaw and measurement row counts differ")
    return sensor.astype(int), yaw, basis


def _select(values, sensor):
    values = np.asarray(values, float)
    if values.ndim < 2 or values.shape[1] != 2:
        raise SGPSSourceError("Expected SGPS pair of sensor units")
    chosen = values[np.arange(len(sensor)), np.maximum(sensor, 0)].copy()
    chosen[sensor < 0] = np.nan
    return chosen


def _map_frame(index, flux, energy, error, valid, advisory, counts, sensor, yaw, product, satellite, energy_basis):
    """Keep native fluxes and QC alongside every derived value."""
    frame = {}
    for i in range(flux.shape[1]):
        frame[f"native_flux_{i}"] = np.where(valid[:, i], flux[:, i], np.nan)
        frame[f"native_energy_MeV_{i}"] = energy[:, i]
        frame[f"native_uncertainty_{i}"] = error[:, i]
        frame[f"native_valid_samples_{i}"] = counts[:, i]
        frame[f"native_advisory_fraction_{i}"] = advisory[:, i]
    clean = np.where(valid, flux, np.nan)
    for feature, target in TARGET_ENERGIES.items():
        mapped = interpolate_energy(energy, clean, target, error)
        frame[feature] = mapped.pop("value")
        for name, values in mapped.items():
            frame[f"{feature}_{name}"] = values
        lo, hi = mapped["lower_index"], mapped["upper_index"]
        good = lo >= 0
        flag = np.full(len(index), np.nan)
        flag[good] = np.maximum(advisory[np.flatnonzero(good), lo[good]], advisory[np.flatnonzero(good), hi[good]])
        frame[f"{feature}_advisory_fraction"] = flag
        frame[f"{feature}_channel_status"] = np.where(np.isfinite(frame[feature]), METHOD, "sgps_invalid_or_nonpositive_adjacent_brackets")
    frame["source_product"], frame["source_satellite"] = product, f"GOES-{satellite}"
    frame["source_detector"] = np.where(sensor == 0, "SGPS-X physical west", np.where(sensor == 1, "SGPS+X physical west", "unknown yaw; rejected"))
    frame["yaw_flag"], frame["energy_coordinate_basis"] = yaw, energy_basis
    frame = pd.DataFrame(frame, index=index)
    frame.attrs["native_valid_points"] = int(valid.sum())
    return frame


def read_l2(path, satellite, min_valid_fraction=0.5):
    """Read current or older SGPS L2, including reprocessed September 2017."""
    with netCDF4.Dataset(path) as dataset:
        time_name = "time" if "time" in dataset.variables else "L2_SciData_TimeStamp"
        if time_name not in dataset.variables:
            raise SGPSSourceError("SGPS L2 lacks timestamps")
        index = _timestamps(dataset[time_name])
        if index.isna().any() or index.has_duplicates:
            raise SGPSSourceError("Invalid or duplicate SGPS L2 timestamps")
        variable = dataset["AvgDiffProtonFlux"]
        flux_all = _array(variable) * _flux_factor(variable)
        n = len(flux_all)
        sensor, yaw, yaw_basis = _physical_west(dataset, n)
        flux = _select(flux_all, sensor)
        energy_all = _energies(dataset["DiffProtonEffectiveEnergy"])
        energy_basis = "native_L2_DiffProtonEffectiveEnergy"
        if not np.isfinite(energy_all).all() or (energy_all <= 0).any():
            # The official reprocessed September-2017 file explicitly marks
            # effective energy as "Not used", but supplies native band bounds.
            lower, upper = _energies(dataset["DiffProtonLowerEnergy"]), _energies(dataset["DiffProtonUpperEnergy"])
            if not (np.isfinite(lower).all() and np.isfinite(upper).all() and (lower > 0).all() and (upper > lower).all()):
                raise SGPSSourceError("No usable native SGPS energy coordinates")
            energy_all = np.sqrt(lower * upper)
            energy_basis = "geometric_mean_of_native_L2_band_bounds_effective_field_unused"
        if energy_all.shape != (2, flux.shape[1]):
            raise SGPSSourceError("Unexpected SGPS effective-energy dimensions")
        energy = energy_all[np.maximum(sensor, 0)]
        error_variable = dataset["AvgDiffProtonFluxUncert"]
        error = _select(_array(error_variable) * _flux_factor(error_variable), sensor)
        counts = _select(_array(dataset["DiffValidL1bSamplesInAvg"]), sensor)
        valid = np.isfinite(flux) & (flux >= 0) & (counts >= 300 * min_valid_fraction) & (sensor[:, None] >= 0)
        # The producer has already removed invalid L1b samples. Modern files
        # additionally tell us if invalid flags were ignored during averaging.
        if "DiffProtonIgnoredL1bDQFs" in dataset.variables:
            ignored = _select(_array(dataset["DiffProtonIgnoredL1bDQFs"]), sensor)
            valid &= np.isfinite(ignored) & (ignored == 0)
        if "ExpectedLUTNotFound" in dataset.variables:
            lut = dataset["ExpectedLUTNotFound"]
            description = str(getattr(lut, "long_name", ""))
            match = re.search(r"([01]) if all input files", description)
            if not match:
                raise SGPSSourceError("Unrecognized SGPS lookup-table flag semantics")
            if float(_array(lut)) != int(match[1]):
                valid[:] = False
        advisory = np.zeros_like(flux)
        for name in ("DiffDQFdtcSum", "DiffDQFoobSum", "DiffDQFerrSum"):
            if name not in dataset.variables:
                raise SGPSSourceError(f"Missing SGPS L2 quality field: {name}")
            advisory = np.maximum(advisory, _select(_array(dataset[name]), sensor) / np.maximum(counts, 1))
        product = "NOAA_SGPS_L2_AVG5M"
        frame = _map_frame(index, flux, energy, error, valid, advisory, counts, sensor, yaw, product, satellite, energy_basis)
        details = {"level": "L2", "satellite": f"GOES-{satellite}", "native_flux_units": str(variable.units), "native_effective_energy_MeV": energy_all.tolist(), "energy_coordinate_basis": energy_basis, "yaw_policy": yaw_basis, "processing": "native 5-minute temperature-corrected AvgDiffProtonFlux; no temporal imputation"}
        return frame, details


def _labels(variable):
    rows = np.ma.filled(variable[:], b" ")
    return [b"".join(row).decode().strip() for row in rows]


def read_l1b(path, satellite, min_valid_fraction=0.5):
    """Genuine 2020 bridge: average valid native 1-s flux, preserving QC.

    L1b lacks the L2 temperature correction and reports nominal energy labels,
    so its energy coordinates are label-bound geometric means, explicitly
    distinguished from L2 calibrated effective energies in every record.
    """
    with netCDF4.Dataset(path) as dataset:
        time_variable = dataset["L1a_SciData_TimeStamp"]
        times = _array(time_variable)
        sensor, yaw, yaw_basis = _physical_west(dataset, len(times))
        selected_times = _select(times, sensor)
        index = _timestamps(time_variable, selected_times)
        fluxes, uncertainties, flags, energy = [], [], [], []
        for telescope in ("T1", "T2", "T3"):
            variable = dataset[f"{telescope}_DifferentialProtonFluxes"]
            fluxes.append(_select(_array(variable) * _flux_factor(variable), sensor))
            variable_error = dataset[f"{telescope}_DifferentialProtonFluxUncertainties"]
            uncertainties.append(_select(_array(variable_error) * _flux_factor(variable_error), sensor))
            flags.append(_select(_array(dataset[f"{telescope}_DifferentialProtonFluxDQFs"]), sensor))
            for label in _labels(dataset[f"energy_{telescope}_label"]):
                match = re.search(r":\s*([\d.]+)\s*-\s*([\d.]+)\s*MeV", label)
                if not match:
                    raise SGPSSourceError(f"Unrecognized L1b energy label: {label}")
                energy.append(np.sqrt(float(match[1]) * float(match[2])))
        flux, error, dqf = np.concatenate(fluxes, axis=1), np.concatenate(uncertainties, axis=1), np.concatenate(flags, axis=1)
        # Bits 1/2 are invalid. Bits 4/8/16 are degraded/advisory, retained
        # with explicit fractions and uncertainty rather than hidden as good.
        valid = np.isfinite(flux) & (flux >= 0) & np.isfinite(dqf) & ((np.nan_to_num(dqf, nan=255).astype(int) & 3) == 0)
        valid &= (sensor >= 0)[:, None] & ~index.isna()[:, None]
        keep = ~index.isna()
        if not keep.any():
            raise SGPSSourceError("L1b has no interpretable timestamps")
        idx = index[keep]
        clean = pd.DataFrame(np.where(valid[keep], flux[keep], np.nan), index=idx)
        if clean.index.has_duplicates:
            raise SGPSSourceError("Duplicate selected-sensor L1b timestamps")
        averaged = clean.resample("5min").mean()
        count = clean.resample("5min").count().to_numpy(float)
        valid_error = valid[keep] & np.isfinite(error[keep]) & (error[keep] >= 0)
        variance = pd.DataFrame(np.where(valid_error, error[keep] ** 2, np.nan), index=idx).resample("5min").sum(min_count=1).to_numpy()
        averaged_error = np.sqrt(variance) / np.maximum(count, 1)
        error_count = pd.DataFrame(valid_error, index=idx).resample("5min").sum().to_numpy()
        averaged_error[error_count != count] = np.nan
        degraded = (np.nan_to_num(dqf[keep], nan=255).astype(int) & 28) != 0
        advisory = pd.DataFrame(np.where(valid[keep], degraded, np.nan), index=idx).resample("5min").mean().to_numpy()
        row_sensor = pd.Series(sensor[keep], index=idx).resample("5min").agg(lambda s: s.iloc[0] if s.nunique() == 1 else -1).to_numpy()
        row_yaw = pd.Series(yaw[keep], index=idx).resample("5min").first().to_numpy()
        good = (count >= 300 * min_valid_fraction) & np.isfinite(averaged.to_numpy()) & (row_sensor >= 0)[:, None]
        frame = _map_frame(averaged.index, averaged.to_numpy(), np.broadcast_to(energy, averaged.shape), averaged_error, good, advisory, count, row_sensor, row_yaw, "NOAA_SGPS_L1B_VALID_1S_TO_5MIN", satellite, "geometric_mean_of_native_L1b_nominal_label_bounds")
        details = {"level": "L1b", "satellite": f"GOES-{satellite}", "native_flux_units": "per keV; converted to per MeV", "nominal_geometric_mean_energy_MeV": energy, "yaw_policy": yaw_basis, "processing": "native valid 1-second measurements averaged in left-labelled 5-minute bins; NO L2 temperature correction", "energy_caveat": "Native L1b labels do not provide versioned calibrated effective energies; nominal label-bound geometric means are used and flagged", "uncertainty_processing": "root-sum-square native 1-second errors / valid sample count, assuming independent samples; correlated systematic uncertainty is not quantified"}
        return frame, details


class SGPSStore:
    """Bounded-RAM daily SGPS adapter with atomic raw/parsed/index caches."""
    def __init__(self, cache_dir="data/cache/sgps", progress=print, *, timeout=120, retries=3, satellites=(16, 18, 19, 17), memory_days=8, min_valid_fraction=0.5):
        if not satellites or any(s not in (16, 17, 18, 19) for s in satellites):
            raise ValueError("Choose GOES satellites 16, 17, 18 and/or 19")
        if not 0 < min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        self.cache_dir, self.progress, self.timeout, self.retries = Path(cache_dir), progress, timeout, retries
        self.satellites, self.memory_days, self.min_valid_fraction = tuple(satellites), max(1, memory_days), min_valid_fraction
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source_records, self._records, self._index_memory, self._day_memory = [], set(), {}, OrderedDict()
        config = json.dumps([CACHE_VERSION, self.satellites, self.min_valid_fraction])
        self._config = hashlib.sha256(config.encode()).hexdigest()[:12]

    def metadata(self, feature):
        if feature not in TARGET_ENERGIES:
            raise KeyError(feature)
        return {"feature": feature, "target_energy_MeV": TARGET_ENERGIES[feature], "units": UNITS, "method": METHOD,
                "quantity": "energy-interpolated differential point-flux estimate; not a native SGPS P4/P5/P6 channel",
                "interpolation": "linear log(flux) versus log(energy) between adjacent positive, quality-valid native energy channels; no extrapolation or temporal filling",
                "calibration_caveat": "Energy matching is not proven equality with the legacy EPS/EPEAD energy response, calibration or background. L1b nominal-energy and L2 calibrated-energy/temperature-correction eras differ.",
                "uncertainty": "Independent-bracket propagation of provided instrument errors only; spectral curvature, correlated calibration and legacy-response mismatch are not quantified",
                "L1b_uncertainty_aggregation": "root-sum-square valid native 1-second errors / sample count assumes independent samples; not a total calibration or systematic uncertainty",
                "advisory_fraction_definition": "L2 maximum of per-type degraded sample counts / valid sample count; L1b fraction with any advisory bit 4/8/16; mapped value uses maximum of its two brackets",
                "quality_policy": f"At least {100*self.min_valid_fraction:g}% valid 1-second samples; hard invalid bits/fills/ambiguous yaw rejected; advisory degraded flags retained and exposed with uncertainty",
                "detector_policy": "physical west: -X upright, +X inverted; ambiguous yaw rejected. Legacy aggregate yaw accepted only when zero/upright",
                "satellite_priority": list(self.satellites), "satellite_policy": "first available native-valid daily product in explicit priority order; not asserted to be NOAA operational primary; no fallback chosen merely to replace a zero log bracket",
                "bin_lookahead_minutes": 5, "timestamp_semantics": "L2 native start-of-average timestamps; L1b end-of-1-second timestamps aggregated into left-labelled 5-minute bins; availability conservatively timestamp+5min",
                "coverage": "Standard archives begin in 2020; separately reprocessed September 2017 is supported. Other unavailable periods stay missing and are reported.", "documentation": DOCUMENTATION}

    def range(self, feature, start, stop):
        self.metadata(feature)
        frame = self._range(start, stop)
        result = frame[feature].copy()
        result.attrs.update(self.metadata(feature))
        return result

    def quality_range(self, feature, start, stop):
        self.metadata(feature)
        frame = self._range(start, stop)
        names = ["channel_status", "uncertainty", "lower_energy", "upper_energy", "lower_flux", "upper_flux", "weight", "advisory_fraction"]
        result = frame[[f"{feature}_{name}" for name in names] + ["source_product", "source_satellite", "source_detector", "yaw_flag", "energy_coordinate_basis"]].rename(columns={f"{feature}_{name}": name for name in names}).copy()
        result["measurement_missing"] = frame[feature].isna()
        result["target_energy_MeV"] = TARGET_ENERGIES[feature]
        return result

    def native_range(self, start, stop):
        """Return retained native flux, energy, uncertainty, coverage and flags."""
        frame = self._range(start, stop)
        return frame[[name for name in frame if name.startswith("native_") or name.startswith("source_") or name in ("yaw_flag", "energy_coordinate_basis")]].copy()

    def _range(self, start, stop):
        start, stop = _utc(start), _utc(stop)
        if stop <= start:
            raise ValueError("stop must be later than start")
        frames = [self._day(day) for day in pd.date_range(start.normalize(), (stop - pd.Timedelta(nanoseconds=1)).normalize(), freq="D")]
        frame = pd.concat(frames).sort_index()
        if frame.index.has_duplicates:
            raise SGPSSourceError("Duplicate combined SGPS timestamps")
        return frame[(frame.index >= start) & (frame.index < stop)]

    def _record(self, record):
        key = json.dumps(record, sort_keys=True, allow_nan=False)
        if key not in self._records:
            self.source_records.append(record)
            self._records.add(key)

    def _empty(self, day):
        index = _grid(day)
        frame = pd.DataFrame(index=index)
        for feature in TARGET_ENERGIES:
            frame[feature] = np.nan
            for field in ("uncertainty", "lower_energy", "upper_energy", "lower_flux", "upper_flux", "weight", "advisory_fraction"):
                frame[f"{feature}_{field}"] = np.nan
            frame[f"{feature}_channel_status"] = "sgps_no_source_coverage"
        frame["source_product"], frame["source_satellite"], frame["source_detector"] = "unavailable", "unavailable", "unavailable"
        frame["yaw_flag"], frame["energy_coordinate_basis"] = np.nan, "unavailable"
        return frame

    def _day(self, day):
        key = day.strftime("%Y-%m-%d")
        if key in self._day_memory:
            self._day_memory.move_to_end(key)
            return self._day_memory[key]
        path = self.cache_dir / "parsed" / self._config / f"{key}.csv"
        manifest = path.with_suffix(".json")
        frame = None
        if path.exists() and manifest.exists():
            try:
                saved = json.loads(manifest.read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest() != saved["parsed_sha256"]:
                    raise ValueError("parsed checksum mismatch")
                frame = pd.read_csv(path, index_col="Time", parse_dates=["Time"], float_precision="round_trip")
                frame.index = pd.to_datetime(frame.index, utc=True)
                if not frame.index.equals(_grid(day)):
                    raise ValueError("incomplete cached SGPS grid")
                frame.index = _grid(day)
                for record in saved["sources"]:
                    self._record(record)
            except (OSError, ValueError, KeyError):
                frame = None
        if frame is None:
            if self.progress:
                self.progress(f"[SGPS] {key}: locating measured differential flux")
            records = []
            for satellite, level, url in self._candidates(day):
                raw_path, sha = self._download_netcdf(url)
                candidate, details = (read_l2 if level == "L2" else read_l1b)(raw_path, satellite, self.min_valid_fraction)
                record = {"url": url, "sha256": sha, "method": METHOD, "targets_MeV": TARGET_ENERGIES, "details": details}
                records.append(record)
                source_valid = candidate.attrs.get("native_valid_points", 0)
                candidate = candidate[(candidate.index >= day) & (candidate.index < day + pd.Timedelta(days=1))]
                if source_valid and not candidate.empty:
                    frame = candidate.reindex(_grid(day))
                    for feature in TARGET_ENERGIES:
                        frame[f"{feature}_channel_status"] = frame[f"{feature}_channel_status"].fillna("sgps_missing_native_observation")
                    for name in ("source_product", "source_satellite", "source_detector", "energy_coordinate_basis"):
                        frame[name] = frame[name].fillna("missing_native_observation")
                    break
            if frame is None:
                frame = self._empty(day)
                records.append({"day": key, "method": METHOD, "source_status": "no_available_native_valid_sgps_product", "reason": "archive index has no daily product or all candidates have invalid native measurements; no synthetic values inserted"})
            for record in records:
                self._record(record)
            content = frame.to_csv(index_label="Time").encode()
            _atomic_bytes(path, content)
            _json(manifest, {"parsed_sha256": hashlib.sha256(content).hexdigest(), "sources": records, "cache_version": CACHE_VERSION})
        self._day_memory[key] = frame
        while len(self._day_memory) > self.memory_days:
            self._day_memory.popitem(last=False)
        return frame

    def _candidates(self, day):
        if day.year == 2017 and day.month == 9 and 16 in self.satellites:
            yield 16, "L2", SPECIAL_2017
            return
        for satellite in self.satellites:
            for level in ("L2", "L1b"):
                directory = f"{ARCHIVE}/goes{satellite}/l2/data/sgps-l2-avg5m/{day:%Y/%m}/" if level == "L2" else f"{ARCHIVE}/goes{satellite}/l1b/seis-l1b-sgps/{day:%Y/%m}/"
                files = self._listing(directory)
                matches = [url for url in files if f"_g{satellite}_d{day:%Y%m%d}_" in url]
                if matches:
                    # Highest numeric version, science preference as tie-break.
                    def version(url):
                        match = re.search(r"_v([\d_-]+)\.nc$", url)
                        return tuple(int(v) for v in re.split(r"[-_]", match[1])) if match else ()
                    yield satellite, level, sorted(matches, key=lambda url: (version(url), "/sci_" in url), reverse=True)[0]

    def _listing(self, url):
        if url in self._index_memory:
            return self._index_memory[url]
        path = self.cache_dir / "indexes" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        if path.exists():
            files = json.loads(path.read_text())["files"]
        else:
            response = self._request(url, allow_absent=True)
            if response is None:
                files = []
            else:
                text = response.content.decode("utf-8")
                if "index of" not in text.lower():
                    raise SGPSSourceError(f"Unexpected NOAA directory response: {url}")
                files = [urljoin(url, name) for name in re.findall(r'href="([^"?]+\.nc)"', text) if "/" not in name]
            _json(path, {"url": url, "files": files, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "status": "available" if response is not None else "archive_directory_not_available"})
        self._index_memory[url] = files
        return files

    def _request(self, url, allow_absent=False):
        error = None
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=(20, self.timeout))
                if allow_absent and response.status_code == 404:
                    return None
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                error = exc
                if self.progress:
                    self.progress(f"[SGPS RETRY {attempt + 1}/{self.retries}] {url}: {exc}")
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 5))
        raise SGPSSourceError(f"SGPS request failed; not converted into an observation gap: {url}: {error}")

    def _download_netcdf(self, url):
        path = self.cache_dir / "raw" / f"{hashlib.sha256(url.encode()).hexdigest()}.nc"
        if not path.exists():
            if self.progress:
                self.progress(f"[SGPS DOWNLOAD] {url.rsplit('/', 1)[-1]}")
            body = self._request(url).content
            if not (body.startswith(b"CDF") or body.startswith(b"\x89HDF\r\n\x1a\n")):
                raise SGPSSourceError(f"NOAA returned a non-netCDF payload: {url}")
            # Validate before publishing raw cache; truncated downloads fail.
            try:
                with netCDF4.Dataset("memory", memory=body):
                    pass
            except (OSError, RuntimeError) as exc:
                raise SGPSSourceError(f"Invalid NOAA netCDF: {url}: {exc}") from exc
            _atomic_bytes(path, body)
        body = path.read_bytes()
        return path, hashlib.sha256(body).hexdigest()

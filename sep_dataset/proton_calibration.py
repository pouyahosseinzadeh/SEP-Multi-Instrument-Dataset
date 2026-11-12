"""Validated, source-scoped estimates of the legacy proton sensor response.

This module never equalizes populations from different years.  It learns a
nonnegative spectral response and background offset from simultaneous
observations, then tests separate held-out quiet and event intervals.  Its
output is a legacy *instrument-equivalent estimate*, not a recovered native
EPEAD observation or proof of homogeneous physical calibration.

Unvalidated products, look directions and satellites fail closed.  A bridge
can be fitted against a previously validated reference estimate, but its
lineage and the weaker validation target are explicit in the artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "sep-proton-response-calibration-v1"
BANK_SCHEMA = "sep-proton-calibration-bank-v1"
FEATURES = ("P4", "P5", "P6")
KEY_FIELDS = ("source_satellite", "source_product", "source_detector", "energy_coordinate_basis")
UNITS = "protons/(cm^2 s sr MeV)"
REFERENCE_QUANTITY = "GOES-15 EPEAD W-labelled UNCOR differential channel-equivalent estimate"
MIN_RESPONSE_ENERGY_MEV = {"P4": 15.0, "P5": 40.0, "P6": 80.0}
L2_BASES = ("geometric_mean_of_native_L2_band_bounds_effective_field_unused",
            "native_L2_DiffProtonEffectiveEnergy")
DEFAULT_GATES = {
    "minimum_training_pairs": 200,
    "minimum_validation_pairs_per_regime": 100,
    "minimum_validation_coverage": 0.95,
    "maximum_quiet_median_relative_bias": 0.20,
    "maximum_quiet_median_absolute_relative_error": 0.35,
    "maximum_event_median_relative_bias": 0.30,
    "maximum_event_median_absolute_relative_error": 0.35,
    "minimum_event_r2": 0.80,
}


class CalibrationError(ValueError):
    """A calibration is invalid, unsupported, or has not passed validation."""


def _key(value):
    if not isinstance(value, dict) or any(not str(value.get(k, "")).strip() for k in KEY_FIELDS):
        raise CalibrationError("A calibration source key needs satellite, product, detector and energy basis")
    return {k: str(value[k]).strip() for k in KEY_FIELDS}


def frame_source_key(frame, supplied=None):
    """Read one explicit source identity; never silently merge source products."""
    observed = {}
    for column in KEY_FIELDS:
        if column in frame:
            values = frame[column].dropna().astype(str).unique()
            if len(values) != 1:
                raise CalibrationError("Calibrate one source group at a time: " + column)
            observed[column] = values[0]
    if supplied is None:
        return _key(observed)
    supplied = _key(supplied)
    if any(supplied[k] != v for k, v in observed.items()):
        raise CalibrationError("Supplied source identity disagrees with native records")
    return supplied


def _columns(frame):
    suffixes = sorted(int(c.rsplit("_", 1)[1]) for c in frame if c.startswith("native_flux_") and c.rsplit("_", 1)[1].isdigit())
    if len(suffixes) < 3 or suffixes != list(range(len(suffixes))):
        raise CalibrationError("Native spectral channels must have contiguous numbered flux columns")
    flux_columns = ["native_flux_{}".format(i) for i in suffixes]
    energy_columns = ["native_energy_MeV_{}".format(i) for i in suffixes]
    if any(c not in frame for c in energy_columns):
        raise CalibrationError("Native calibrated energy coordinates are required")
    return flux_columns, energy_columns


def native_spectral_matrix(frame, reference_grid=None, grid_rtol=0.005):
    """Map adjacent native samples to a fixed grid, preserving valid zeros.

    Flux is linear in log-energy.  Unlike a log-flux mapping, a native zero is
    not discarded.  Invalid adjacent channels are never skipped, and there is
    no extrapolation.  The narrow energy-grid tolerance protects the source
    definition, rather than asserting that different sensors are equivalent.
    """
    flux_columns, energy_columns = _columns(frame)
    flux = frame[flux_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    energies = frame[energy_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if reference_grid is None:
        valid_energy = np.isfinite(energies).all(axis=1) & (energies > 0).all(axis=1) & (np.diff(energies, axis=1) > 0).all(axis=1)
        if not valid_energy.any():
            raise CalibrationError("No valid native energy grid")
        reference_grid = np.median(energies[valid_energy], axis=0)
    grid = np.asarray(reference_grid, dtype=float)
    if grid.ndim != 1 or len(grid) != len(flux_columns) or not np.isfinite(grid).all() or (grid <= 0).any() or (np.diff(grid) <= 0).any():
        raise CalibrationError("Invalid reference energy grid")
    result = np.full((len(frame), len(grid)), np.nan)
    compatible = np.isfinite(energies).all(axis=1) & (energies > 0).all(axis=1) & (np.diff(energies, axis=1) > 0).all(axis=1)
    compatible &= (np.abs(energies-grid) <= grid_rtol * grid).all(axis=1)
    flux[(flux < 0) | (~np.isfinite(flux)) | (flux >= 1e29)] = np.nan
    for row in np.flatnonzero(compatible):
        e, values = energies[row], flux[row]
        # Tiny metadata-rounding differences at the endpoints are snapped to
        # the native endpoint; no flux outside the measured range is inferred.
        targets = np.clip(grid, e[0], e[-1])
        for column, target in enumerate(targets):
            hi = min(int(np.searchsorted(e, target, side="left")), len(e)-1)
            lo = hi if np.isclose(e[hi], target, rtol=1e-10, atol=0) else hi-1
            if lo < 0 or not (np.isfinite(values[lo]) and np.isfinite(values[hi])):
                continue
            weight = 0.0 if hi == lo else np.log(target/e[lo]) / np.log(e[hi]/e[lo])
            result[row, column] = (1-weight)*values[lo] + weight*values[hi]
    return result, grid, compatible


def _mask(value, size, name):
    result = np.asarray(value)
    if result.shape != (size,) or result.dtype != bool:
        raise CalibrationError(name + " must be a boolean row mask")
    return result.copy()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def artifact_digest(artifact):
    value = {k: v for k, v in artifact.items() if k != "calibration_id"}
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _seal(artifact):
    artifact = _jsonable(artifact)
    artifact["calibration_id"] = artifact_digest(artifact)
    return artifact


def _check_artifact(artifact, require_pass=True):
    if artifact.get("schema") != SCHEMA or artifact.get("calibration_id") != artifact_digest(artifact):
        raise CalibrationError("Calibration schema or content hash is invalid")
    if require_pass and artifact.get("status") != "passed":
        raise CalibrationError("Calibration did not pass independent quiet/event validation")
    if artifact.get("units") != UNITS or set(artifact.get("models", {})) != set(FEATURES):
        raise CalibrationError("Unsupported calibration output definition")
    for model in artifact["models"].values():
        coefficient = np.asarray(model.get("coefficients", []), dtype=float)
        scales = np.asarray(model.get("feature_scales", []), dtype=float)
        if len(coefficient) != len(artifact["reference_grid_MeV"]) or coefficient.shape != scales.shape:
            raise CalibrationError("Malformed response coefficients")
        if not np.isfinite(coefficient).all() or (coefficient < 0).any() or not np.isfinite(scales).all() or (scales <= 0).any() or not np.isfinite(model["intercept"]) or model["intercept"] < 0:
            raise CalibrationError("Response coefficients and background must be nonnegative and finite")


def _metrics(observed, predicted, requested):
    valid = requested & np.isfinite(observed) & (observed > 0) & np.isfinite(predicted) & (predicted >= 0)
    n_requested, n = int(requested.sum()), int(valid.sum())
    result = {"requested_rows": n_requested, "valid_pairs": n,
              "coverage": n/n_requested if n_requested else 0.0,
              "median_relative_bias": None, "median_absolute_relative_error": None,
              "r2": None, "observed_median": None, "predicted_median": None,
              "absolute_error_p95": None}
    if n:
        y, p = observed[valid], predicted[valid]
        ratio = p/y
        variance = float(np.sum((y-y.mean())**2))
        result.update(median_relative_bias=float(np.median(ratio-1)),
                      median_absolute_relative_error=float(np.median(np.abs(ratio-1))),
                      r2=1-float(np.sum((p-y)**2))/variance if variance > 0 else None,
                      observed_median=float(np.median(y)), predicted_median=float(np.median(p)),
                      absolute_error_p95=float(np.percentile(np.abs(p-y), 95)))
    return result


def _gate(metrics, regime, gates):
    reasons = []
    if metrics["valid_pairs"] < gates["minimum_validation_pairs_per_regime"]:
        reasons.append(regime + ": too few held-out pairs")
    if metrics["coverage"] < gates["minimum_validation_coverage"]:
        reasons.append(regime + ": held-out coverage below requirement")
    bias, error = metrics["median_relative_bias"], metrics["median_absolute_relative_error"]
    if bias is None or abs(bias) > gates["maximum_{}_median_relative_bias".format(regime)]:
        reasons.append(regime + ": median relative bias exceeds requirement")
    if error is None or error > gates["maximum_{}_median_absolute_relative_error".format(regime)]:
        reasons.append(regime + ": median absolute relative error exceeds requirement")
    if regime == "event" and (metrics["r2"] is None or metrics["r2"] < gates["minimum_event_r2"]):
        reasons.append("event: R2 below requirement")
    return reasons


def _fit_response(x, y):
    from scipy.optimize import nnls
    scale = np.percentile(x, 90, axis=0)
    scale[scale <= 0] = 1.0
    design = np.column_stack([np.ones(len(x)), x/scale])
    floor = max(float(np.percentile(y, 10)), np.finfo(float).tiny)
    weight = 1/np.maximum(y, floor)
    for _ in range(4):
        coefficient, _ = nnls(design*weight[:, None], y*weight, maxiter=1000)
        residual = (design@coefficient-y)/np.maximum(y, floor)
        weight = 1/np.maximum(y, floor) * np.sqrt(np.minimum(1.0, 0.3/np.maximum(np.abs(residual), 1e-12)))
    return coefficient, scale, floor


def fit_calibration(native_frame, targets, train_mask, validation_mask, *, source_key=None,
                    validation_regimes=None, provenance=None, reference_kind="observed_legacy",
                    parent_artifact=None, gates=None, reference_grid=None, grid_rtol=0.005):
    """Fit on simultaneous training observations; validate on disjoint blocks.

    ``validation_regimes`` must explicitly name quiet and event masks drawn
    entirely from validation rows. Coverage is measured against ALL requested
    rows in each regime, including missing spectra and missing targets.
    """
    if not isinstance(native_frame.index, pd.DatetimeIndex) or native_frame.index.has_duplicates or native_frame.index.isna().any() or native_frame.index.tz is None:
        raise CalibrationError("Unique, valid timezone-aware native timestamps are required")
    if not native_frame.index.equals(targets.index) or not set(FEATURES).issubset(targets):
        raise CalibrationError("Legacy/reference targets must be aligned to native timestamps")
    key = frame_source_key(native_frame, source_key)
    n = len(native_frame)
    train, validation = _mask(train_mask, n, "train_mask"), _mask(validation_mask, n, "validation_mask")
    if (train & validation).any():
        raise CalibrationError("Training and held-out rows overlap")
    if set(validation_regimes or {}) != {"quiet", "event"}:
        raise CalibrationError("Separate held-out quiet and event masks are required")
    regimes = {k: _mask(v, n, k) for k, v in validation_regimes.items()}
    if any((mask & ~validation).any() for mask in regimes.values()) or (regimes["quiet"] & regimes["event"]).any():
        raise CalibrationError("Validation regimes must be disjoint subsets of held-out rows")
    if reference_kind not in ("observed_legacy", "calibrated_reference_bridge"):
        raise CalibrationError("Unknown reference target kind")
    if reference_kind == "calibrated_reference_bridge":
        if parent_artifact is None:
            raise CalibrationError("A bridge needs its validated reference parent artifact")
        _check_artifact(parent_artifact)
    elif parent_artifact is not None:
        raise CalibrationError("Observed legacy calibration must not claim a modeled parent")
    policy = dict(DEFAULT_GATES)
    if gates:
        if set(gates)-set(policy):
            raise CalibrationError("Unknown validation gate")
        policy.update(gates)
    if not 0 < policy["minimum_validation_coverage"] <= 1 or not 0 < grid_rtol <= 0.05:
        raise CalibrationError("Invalid validation coverage or energy-grid tolerance")
    if policy["minimum_training_pairs"] < 3 or policy["minimum_validation_pairs_per_regime"] < 3:
        raise CalibrationError("Calibration requires multiple independent held-out observations")
    # Derive the grid from training energy metadata only.  Test spectra and
    # targets never enter coefficient fitting or selection of loss weights.
    _, energy_columns = _columns(native_frame)
    if reference_grid is None:
        energy = native_frame.loc[train, energy_columns].to_numpy(dtype=float)
        good_energy = np.isfinite(energy).all(axis=1) & (energy > 0).all(axis=1)
        if not good_energy.any():
            raise CalibrationError("No training energy coordinates")
        reference_grid = np.median(energy[good_energy], axis=0)
    x, grid, grid_ok = native_spectral_matrix(native_frame, reference_grid, grid_rtol)
    models = {}
    reasons = []
    for feature in FEATURES:
        y = pd.to_numeric(targets[feature], errors="coerce").to_numpy(dtype=float)
        required = np.flatnonzero(grid >= MIN_RESPONSE_ENERGY_MEV[feature])
        if len(required) < 2:
            raise CalibrationError("Insufficient high-energy native channels for " + feature)
        spectra_valid = np.isfinite(x[:, required]).all(axis=1) & (x[:, required] >= 0).all(axis=1) & grid_ok
        valid_train = train & spectra_valid & np.isfinite(y) & (y > 0) & (y < 1e29)
        if valid_train.sum() < policy["minimum_training_pairs"]:
            raise CalibrationError("Too few simultaneous training observations for " + feature)
        fitted, fitted_scales, floor = _fit_response(x[valid_train][:, required], y[valid_train])
        coefficients = np.zeros(len(grid)+1)
        scales = np.ones(len(grid))
        coefficients[0] = fitted[0]
        coefficients[required+1] = fitted[1:]
        scales[required] = fitted_scales
        prediction = np.full(n, np.nan)
        prediction[spectra_valid] = fitted[0] + (x[spectra_valid][:, required]/fitted_scales)@fitted[1:]
        checks = {regime: _metrics(y, prediction, mask) for regime, mask in regimes.items()}
        failures = [reason for regime, values in checks.items() for reason in _gate(values, regime, policy)]
        reasons.extend(feature + ": " + value for value in failures)
        # Bounds describe observed training AND independent validation spectra.
        # Extrapolation beyond this tested envelope is not silently accepted.
        supported = (train | validation) & spectra_valid & np.isfinite(y) & (y > 0)
        upper = np.zeros(len(grid))
        upper[required] = np.max(x[supported][:, required], axis=0)
        active = np.flatnonzero(coefficients[1:] > max(float(np.max(coefficients[1:])), 1e-30)*1e-9)
        models[feature] = {
            "intercept": coefficients[0], "coefficients": coefficients[1:], "feature_scales": scales,
            "active_channels": active.tolist(), "input_upper_bounds": upper,
            "required_channels": required.tolist(), "minimum_response_energy_MeV": MIN_RESPONSE_ENERGY_MEV[feature],
            "training_pairs": int(valid_train.sum()), "training_requested_rows": int(train.sum()),
            "training_relative_error_floor": floor, "validation": checks,
            "passed": not failures, "failure_reasons": failures,
        }
    bases = [key["energy_coordinate_basis"]]
    if key["source_satellite"] == "GOES-16" and key["source_product"] == "NOAA_SGPS_L2_AVG5M" and key["energy_coordinate_basis"] in L2_BASES:
        bases = list(L2_BASES)
    arrays = [native_frame.index.asi8.tobytes(), np.asarray(x, dtype="<f8").tobytes(),
              targets[list(FEATURES)].to_numpy(dtype="<f8").tobytes(), train.tobytes(), validation.tobytes()]
    fingerprint = hashlib.sha256(b"".join(arrays)).hexdigest()
    evidence = dict(provenance or {})
    evidence.setdefault("calibration_dates", sorted(set(native_frame.index[train | validation].tz_convert("UTC").strftime("%Y-%m-%d"))))
    artifact = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not reasons else "failed", "failure_reasons": reasons,
        "reference_quantity": REFERENCE_QUANTITY, "units": UNITS,
        "reference_kind": reference_kind, "reference_grid_MeV": grid,
        "models": models, "validation_gates": policy, "training_fingerprint": fingerprint,
        "applicability": {"source_key": key, "allowed_energy_bases": bases, "energy_grid_rtol": grid_rtol,
                          "numeric_energy_grid_required": True, "allow_extrapolation": False},
        "training_period": [str(native_frame.index[train].min()), str(native_frame.index[train].max())],
        "heldout_period": [str(native_frame.index[validation].min()), str(native_frame.index[validation].max())],
        "method": "Nonnegative response from native channels at/above the legacy nominal lower band energy plus nonnegative intercept; four relative-Huber NNLS fits; no population mean matching",
        "provenance": evidence,
        "lineage": list(parent_artifact.get("lineage", [])) + [parent_artifact["calibration_id"]] if parent_artifact else [],
        "validation_target_note": "Held-out targets are reference-model estimates, not new legacy observations" if parent_artifact else "Held-out targets are simultaneous native legacy W-labelled UNCOR observations",
        "limitations": ["Estimated legacy-instrument-equivalent values, not native EPEAD measurements",
                        "Validation is limited to the stated simultaneous products, source key, energy grid and tested spectral envelope",
                        "Quiet background matching does not establish calibrated physical interplanetary proton flux",
                        "A source bridge inherits reference-model errors; held-out agreement is not independent legacy validation",
                        "Interpolation, later gap filling and future classifier scores do not establish instrument calibration"],
    }
    return _seal(artifact)


def _source_supported(key, artifact):
    app = artifact["applicability"]
    expected = app["source_key"]
    return all(key[k] == expected[k] for k in KEY_FIELDS if k != "energy_coordinate_basis") and key["energy_coordinate_basis"] in app["allowed_energy_bases"]


def apply_calibration(native_frame, artifact, *, source_key=None, strict_source=True):
    """Apply only a passed, source-compatible response inside its tested range."""
    _check_artifact(artifact)
    key = frame_source_key(native_frame, source_key)
    supported = _source_supported(key, artifact)
    if not supported and strict_source:
        raise CalibrationError("No validated calibration for source " + json.dumps(key, sort_keys=True))
    result = pd.DataFrame(index=native_frame.index)
    if supported:
        app = artifact["applicability"]
        x, _, grid_ok = native_spectral_matrix(native_frame, artifact["reference_grid_MeV"], app["energy_grid_rtol"])
    for feature in FEATURES:
        value = np.full(len(native_frame), np.nan)
        status = np.full(len(native_frame), "unsupported_source", dtype=object)
        if supported:
            model = artifact["models"][feature]
            required = np.asarray(model.get("required_channels", list(range(x.shape[1]))), dtype=int)
            finite = np.isfinite(x[:, required]).all(axis=1) & (x[:, required] >= 0).all(axis=1) & grid_ok
            bounds = np.asarray(model["input_upper_bounds"], dtype=float)
            active = np.asarray(model["active_channels"], dtype=int)
            in_domain = (x[:, active] <= bounds[active]*(1+1e-8)+1e-14).all(axis=1) if len(active) else np.ones(len(x), dtype=bool)
            good = finite & in_domain
            value[good] = model["intercept"] + (x[good][:, required]/np.asarray(model["feature_scales"])[required])@np.asarray(model["coefficients"])[required]
            status[:] = "missing_or_invalid_native_spectrum"
            status[~grid_ok] = "unsupported_energy_grid"
            status[finite & ~in_domain] = "outside_validated_spectral_range"
            status[good] = "validated_legacy_equivalent_estimate"
        result[feature] = value
        result[feature + "_calibration_status"] = status
    result["calibration_id"] = artifact["calibration_id"]
    result["calibration_reference_kind"] = artifact["reference_kind"]
    result.attrs.update({"calibration_id": artifact["calibration_id"], "units": UNITS,
                         "quantity": REFERENCE_QUANTITY, "source_key": key})
    return result


def save_calibration(path, artifact):
    """Atomically save the complete artifact, including failed-fit evidence."""
    _check_artifact(artifact, require_pass=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def load_calibration(path, require_pass=True):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_artifact(artifact, require_pass=require_pass)
    return artifact


def load_calibration_bank(path):
    """Validate a bank and all parent-model lineage before applying estimates."""
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    if bank.get("schema") != BANK_SCHEMA or not bank.get("artifacts"):
        raise CalibrationError("No validated proton calibration bank")
    ids = set()
    for artifact in bank["artifacts"]:
        _check_artifact(artifact)
        if artifact["calibration_id"] in ids:
            raise CalibrationError("Duplicate calibration artifact in bank")
        ids.add(artifact["calibration_id"])
    for artifact in bank["artifacts"]:
        if any(parent not in ids for parent in artifact.get("lineage", [])):
            raise CalibrationError("Calibration bank lacks a required validated reference parent")
    return bank


def apply_calibration_bank(native_frame, bank, *, strict_source=True):
    """Apply the unique validated model for each actual native source group.

    Unsupported groups are explicit missing estimates when ``strict_source``
    is false; callers must report their coverage rather than filling from an
    unrelated source model or silently selecting the closest satellite.
    """
    if bank.get("schema") != BANK_SCHEMA:
        raise CalibrationError("Unsupported calibration bank schema")
    for artifact in bank.get("artifacts", []):
        _check_artifact(artifact)
    if any(column not in native_frame for column in KEY_FIELDS):
        raise CalibrationError("Native audit lacks source identity fields")
    if native_frame.index.has_duplicates:
        raise CalibrationError("Resolve duplicate native timestamps before calibration")
    pieces = []
    for _, group in native_frame.groupby(list(KEY_FIELDS), dropna=False, sort=False):
        try:
            key = frame_source_key(group)
        except CalibrationError:
            key = None
        candidates = [a for a in bank.get("artifacts", []) if key is not None and _source_supported(key, a)]
        if len(candidates) > 1:
            raise CalibrationError("Ambiguous calibration bank for one native source")
        if candidates:
            pieces.append(apply_calibration(group, candidates[0]))
        elif strict_source:
            raise CalibrationError("No validated model in bank for source " + json.dumps(key, sort_keys=True))
        else:
            result = pd.DataFrame(index=group.index)
            for feature in FEATURES:
                result[feature] = np.nan
                result[feature+"_calibration_status"] = "unsupported_source"
            result["calibration_id"] = ""
            result["calibration_reference_kind"] = "unavailable"
            pieces.append(result)
    if not pieces:
        return pd.DataFrame(index=native_frame.index)
    return pd.concat(pieces).reindex(native_frame.index)


def validation_table(artifact):
    """A compact, machine-readable fit audit without concealed missing pairs."""
    rows = []
    for feature, model in artifact["models"].items():
        for regime, metrics in model["validation"].items():
            rows.append({"feature": feature, "regime": regime, "calibration_passed": artifact["status"] == "passed",
                         "reference_kind": artifact["reference_kind"], **metrics})
    return pd.DataFrame(rows)


def fit_official_2017(native_netcdf, legacy_csv, *, provenance=None, gates=None):
    """Reproduce the reference fit from the public simultaneous NOAA files.

    Train: September 1–9. Held-out event: September 10–16. Held-out quiet:
    September 20–30. Distinct events/time blocks, never random interleaving.
    """
    from .sgps_sources import read_l2
    native, details = read_l2(native_netcdf, 16)
    text = Path(legacy_csv).read_text(encoding="utf-8")
    marker = text.find("time_tag,")
    if marker < 0:
        raise CalibrationError("Legacy NOAA CSV lacks the time_tag data header")
    legacy = pd.read_csv(io.StringIO(text[marker:]))
    legacy.index = pd.to_datetime(legacy["time_tag"], utc=True)
    if legacy.index.has_duplicates:
        raise CalibrationError("Duplicate legacy calibration timestamps")
    legacy = legacy.reindex(native.index)
    target = pd.DataFrame(index=native.index)
    for feature in FEATURES:
        values = pd.to_numeric(legacy[feature+"W_UNCOR_FLUX"], errors="coerce")
        valid = values.gt(0) & values.lt(1e29) & legacy[feature+"W_QUAL_FLAG"].eq(0)
        target[feature] = values.where(valid)
    before = lambda value: np.asarray(native.index < pd.Timestamp(value, tz="UTC"))
    after = lambda value: np.asarray(native.index >= pd.Timestamp(value, tz="UTC"))
    train = before("2017-09-10")
    quiet = after("2017-09-20")
    event = after("2017-09-10") & before("2017-09-17")
    evidence = dict(provenance or {})
    evidence.update({"native_netcdf_sha256": hashlib.sha256(Path(native_netcdf).read_bytes()).hexdigest(),
                     "legacy_csv_sha256": hashlib.sha256(Path(legacy_csv).read_bytes()).hexdigest(),
                     "native_reader_details": details,
                     "training_definition": "September 1–9, 2017, simultaneous native SGPS and GOES15 W_UNCOR",
                     "heldout_definition": "Event September 10–16; quiet September 20–30, 2017",
                     "calibration_dates": sorted(set(native.index.tz_convert("UTC").strftime("%Y-%m-%d")))})
    return fit_calibration(native, target, train, quiet | event,
                           validation_regimes={"quiet": quiet, "event": event},
                           provenance=evidence, gates=gates)

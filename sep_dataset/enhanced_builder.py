"""Preserve the ten-file release and create a twelve-feature research extension.

P4/P5/P6 after the legacy cutoff are explicitly energy-matched SGPS estimates,
not renewed EPEAD observations or an assertion of identical detector response.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .compatible_baseline import FEATURES as LEGACY_FEATURES
from .compatible_builder import (MISSING, STEPS, _atomic_csv, _atomic_json,
                                 file_sha256, validate_dataset, window_index)
from .compatible_sources import LEGACY_END

MAGNETIC_FEATURES = ("USFLUXL", "R_VALUE")
FEATURES = tuple(LEGACY_FEATURES) + MAGNETIC_FEATURES
VERSION = "twelve-features-sgps-v1"


def feature_columns(feature: str) -> list[str]:
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature}")
    return [f"{feature}_{i}" for i in range(STEPS, 0, -1)]


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    shutil.copyfile(source, part)
    os.replace(part, target)


def _read_quality(path: Path, values: np.ndarray) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing legacy quality audit: {path}. Validate the original build first.")
    q = pd.read_csv(path, dtype={"point_status": str, "causality_mask": str})
    if len(q) != len(values):
        raise ValueError(f"Legacy quality row count mismatch: {path}")
    return q


def _counts(status: str) -> dict:
    return {"observed_points": status.count("O"), "interpolated_points": status.count("I"),
            "edge_filled_points": status.count("E"), "median_filled_points": status.count("M"),
            "derived_points": status.count("D"), "held_points": status.count("H"),
            "unknown_quality_points": status.count("U"), "structural_missing_points": status.count("S"),
            "missing_points_after": status.count("S") + status.count("G")}


def _proton_overlay(feature: str, frame: pd.DataFrame, quality: pd.DataFrame, store, audit_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = feature_columns(feature)
    values = frame[columns].to_numpy(dtype=float, copy=True)
    onsets = pd.to_datetime(frame.FlrOnset, utc=True, format="mixed")
    eligible = np.flatnonzero(onsets > LEGACY_END)
    quality = quality.copy()
    if not len(eligible):
        return frame, quality
    months = onsets.dt.strftime("%Y-%m")
    lookahead = pd.Timedelta(minutes=store.metadata(feature).get("bin_lookahead_minutes", 5))
    for month in months.iloc[eligible].unique():
        rows = eligible[months.iloc[eligible].eq(month).to_numpy()]
        low = max(window_index(onsets.iloc[rows[0]])[0], LEGACY_END)
        high = window_index(onsets.iloc[rows[-1]])[-1] + pd.Timedelta(minutes=5)
        if high <= low:
            continue
        print(f"[CONTINUE] {feature} {month}: {len(rows)} flare windows", flush=True)
        series = store.range(feature, low, high)
        source_quality = store.quality_range(feature, low, high)
        if audit_dir is not None:
            details = source_quality.copy()
            details["derived_flux"] = series
            details["measurement_available_at"] = details.index + lookahead
            details.index.name = "Time"
            _atomic_csv(audit_dir / f"{feature}_{month}.csv", details.reset_index())
            native_path = audit_dir / f"native_sgps_{month}.csv"
            if hasattr(store, "native_range") and (feature == "P4" or not native_path.exists()):
                native = store.native_range(low, high)
                native.index.name = "Time"
                _atomic_csv(native_path, native.reset_index())
        for row in rows:
            grid = window_index(onsets.iloc[row])
            replace = (grid >= LEGACY_END) & ((values[row] == MISSING) | ~np.isfinite(values[row]))
            if not replace.any():
                continue
            candidate = series.reindex(grid).to_numpy(dtype=float)
            candidate[~np.isfinite(candidate) | (candidate < 0)] = np.nan
            excluded = grid + lookahead > onsets.iloc[row]
            candidate[excluded] = np.nan
            status = np.array(list(str(quality.iloc[row].point_status)), dtype="U1")
            if len(status) != STEPS:
                raise ValueError(f"Invalid original quality mask for {feature}, row {row}")
            status[replace] = "S"
            # Limited causal holds make a trailing bin usable without admitting
            # measurements extending beyond onset. Never fill long gaps/years.
            last = None
            last_time = None
            for j in range(STEPS):
                if np.isfinite(candidate[j]):
                    last, last_time = candidate[j], grid[j]
                    if replace[j]:
                        values[row, j], status[j] = candidate[j], "D"
                elif replace[j] and last_time is not None and grid[j] - last_time <= pd.Timedelta(minutes=10):
                    values[row, j], status[j] = last, "H"
            mask = "".join(status)
            counts = _counts(mask)
            for key, value in counts.items():
                quality.loc[row, key] = value
            if "U" in mask:
                # The source historical audit remains unknown, not zero.
                for key in ("observed_points", "interpolated_points", "edge_filled_points", "median_filled_points"):
                    quality.loc[row, key] = np.nan
            q = source_quality.reindex(grid)
            products = q.get("source_product", pd.Series(dtype=str)).dropna().astype(str).unique()
            satellites = q.get("source_satellite", pd.Series(dtype=str)).dropna().astype(str).unique()
            quality.loc[row, "point_status"] = mask
            quality.loc[row, "legacy_channel_missing_points"] = int(replace.sum())
            quality.loc[row, "energy_derived_points"] = int(np.isin(status, ["D", "H"]).sum())
            quality.loc[row, "source_era"] = " | ".join(sorted(products)) or "SGPS_missing_observations"
            quality.loc[row, "source_satellites"] = " | ".join(sorted(satellites)) or "unavailable"
            quality.loc[row, "source_method"] = "SGPS_log_energy_log_flux_interpolation; causal_hold_max_10min; no_cross_instrument_calibration_claim"
            quality.loc[row, "unavailable_channel"] = bool(np.isin(status, ["S", "G"]).any())
            quality.loc[row, "causality_mask"] = "".join(np.where(excluded & replace, "1", "0"))
            quality.loc[row, "causality_excluded_points"] = int((excluded & replace).sum())
    result = frame.copy()
    result[columns] = np.where(np.isfinite(values), values, MISSING)
    return result, quality


def _magnetic_feature(feature: str, metadata: pd.DataFrame, store) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = np.full((len(metadata), STEPS), MISSING, dtype=float)
    audits = []
    for i, row in metadata.iterrows():
        if i % 100 == 0:
            print(f"[MAGNETIC] {feature}: {i}/{len(metadata)} samples", flush=True)
        series, q = store.window(feature, row.FlrOnset, row.get("noaa_active_region"))
        grid = window_index(row.FlrOnset)
        series = series.reindex(grid)
        q = q.reindex(grid)
        data = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(data) & (data != MISSING)
        if feature == "USFLUXL":
            good &= data >= 0
        status = q.get("point_status", pd.Series("G", index=grid)).fillna("G").astype(str).to_numpy()
        # Missing/ambiguous-region/QC causes belong in reason columns; the
        # compact common mask uses G consistently for all remaining gaps.
        status = np.where(good, np.where(np.isin(status, ["O", "H"]), status, "H"), "G")
        values[i] = np.where(good, data, MISSING)
        mask = "".join(status)
        audit = {"row_index": int(i), "sample_id": f"sample-{i}", "origin": "magnetic_archive",
                 "point_status": mask, "causality_mask": "0" * STEPS, "causality_excluded_points": 0,
                 "unavailable_channel": bool((~good).any()), **_counts(mask),
                 "source_method": "native_region_match; backward_asof; instrument_integration_guard; bounded_hold; no_median_fill",
                 "source_era": " | ".join(sorted(q.get("source_product", pd.Series(dtype=str)).dropna().astype(str).unique())),
                 "source_satellites": " | ".join(sorted(q.get("source_satellite", pd.Series(dtype=str)).dropna().astype(str).unique()))}
        for name in ("match_status", "reason", "source_era"):
            if name in q:
                audit["native_" + name] = " | ".join(sorted(q[name].dropna().astype(str).unique()))
        if "age_minutes" in q and good.any():
            audit["maximum_observation_age_minutes"] = float(pd.to_numeric(q.loc[good, "age_minutes"], errors="coerce").max())
        if "native_observation_time" in q and good.any():
            native_times = pd.to_datetime(q.loc[good, "native_observation_time"], utc=True, errors="coerce")
            audit["first_native_observation_time"] = str(native_times.min())
            audit["last_native_observation_time"] = str(native_times.max())
        if "quality_nonzero" in q:
            audit["points_with_native_advisory_quality"] = int(q.loc[good, "quality_nonzero"].fillna(False).sum())
        if "region_count" in q:
            audit["multi_region_patch_points"] = int(pd.to_numeric(q.loc[good, "region_count"], errors="coerce").gt(1).sum())
        if "matched_patch" in q:
            audit["matched_native_patches"] = " | ".join(sorted(q.loc[good, "matched_patch"].dropna().astype(str).unique()))
        if "matched_noaa_ars" in q:
            audit["matched_noaa_membership"] = " | ".join(sorted(q.loc[good, "matched_noaa_ars"].dropna().astype(str).unique()))
        if "noaa_identifier_expanded" in q:
            audit["points_with_expanded_noaa_identifier"] = int(q.loc[good, "noaa_identifier_expanded"].fillna(False).sum())
        audits.append(audit)
    if not np.any(values != MISSING):
        raise RuntimeError(f"No valid {feature} observations matched any sample. Check magnetic source access, NOAA active-region IDs and quality reports; refusing an entirely empty new feature.")
    return pd.concat([metadata.reset_index(drop=True), pd.DataFrame(values, columns=feature_columns(feature))], axis=1), pd.DataFrame(audits)


def _summary(feature: str, quality: pd.DataFrame) -> dict:
    masks = quality.point_status.astype(str)
    total = len(quality) * STEPS
    counts = {code: int(masks.str.count(code).sum()) for code in "OIEMDHSGU"}
    return {"feature": feature, "samples": len(quality), "missing_fraction_after": (counts["S"] + counts["G"]) / total,
            "derived_fraction": counts["D"] / total, "held_fraction": counts["H"] / total,
            "unknown_quality_fraction": counts["U"] / total,
            "point_counts": counts, "whole_window_imputed_count": int(masks.eq("M" * STEPS).sum()),
            "unavailable_windows": int(quality.missing_points_after.gt(0).sum())}


def enhance_dataset(legacy_dir: str | Path = "data/processed/compatible_1998_2025", *,
                    outdir: str | Path = "data/processed/enhanced_1998_2025",
                    cache_dir: str | Path = "data/cache/enhanced", magnetic_csv: str | Path | None = None,
                    verify_legacy: bool = True, sgps_store=None, magnetic_store=None) -> dict:
    legacy_dir, outdir = Path(legacy_dir), Path(outdir)
    if legacy_dir.resolve() == outdir.resolve():
        raise ValueError("Enhanced outputs need their own directory to preserve original files")
    if verify_legacy:
        validate_dataset(legacy_dir)
    original_manifest = json.loads((legacy_dir / "dataset_manifest.json").read_text())
    entries = {e["feature"]: e for e in original_manifest["feature_files"]}
    if list(entries) != list(LEGACY_FEATURES):
        raise ValueError("The input must be a complete original ten-feature build")
    source = legacy_dir / entries["Vx"]["filename"]
    columns = list(pd.read_csv(source, nrows=0).columns[:52])
    metadata = pd.read_csv(source, usecols=columns, float_precision="round_trip", low_memory=False)
    if len(columns) != 52 or not {"FlrOnset", "event_type", "noaa_active_region"}.issubset(metadata):
        raise ValueError("Expected original 52-column sample metadata")
    sample_audit = pd.read_csv(legacy_dir / "quality" / "Vx_quality.csv", usecols=["row_index", "sample_id"], dtype={"sample_id": str})
    if sample_audit.row_index.tolist() != list(range(len(metadata))):
        raise ValueError("Original sample audit is not aligned with metadata")
    end = original_manifest["config"]["end_date_inclusive"]
    config = {"version": VERSION, "legacy_config_sha256": hashlib.sha256(json.dumps(original_manifest["config"], sort_keys=True).encode()).hexdigest(),
              "input_files": {f: e["sha256"] for f, e in entries.items()}, "features": list(FEATURES),
              "input_quality_files": {f: file_sha256(legacy_dir / "quality" / f"{f}_quality.csv") for f in LEGACY_FEATURES},
              "end_date_inclusive": end, "magnetic_csv_sha256": file_sha256(magnetic_csv) if magnetic_csv else None,
              "proton_method": "SGPS energy-matched proxy at 30.6/63.1/165 MeV; not response-equivalent EPEAD",
              "magnetic_method": "native SMARP/SHARP; unique-patch NOAA membership; multi-region patches flagged; backward sampling with integration guard"}
    outdir.mkdir(parents=True, exist_ok=True)
    config_path, state_path = outdir / "build_config.json", outdir / "build_status.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Enhanced output belongs to another configuration/input; choose a new --outdir")
    _atomic_json(config_path, config)
    _atomic_json(state_path, {"state": "running", "config": config})
    try:
        if sgps_store is None:
            from .sgps_sources import SGPSStore
            sgps_store = SGPSStore(Path(cache_dir) / "sgps")
        if magnetic_store is None:
            from .magnetic_sources import MagneticStore
            magnetic_store = MagneticStore(Path(cache_dir) / "magnetic", allow_multi_region_patch=True, **({"native_csv": magnetic_csv} if magnetic_csv else {}))
        results, summaries = [], []
        for feature in FEATURES:
            name = f"1998_{pd.Timestamp(end).year}_MEMSEP_dataset_with_{feature}_filled.csv"
            target, audit_path = outdir / name, outdir / "quality" / f"{feature}_quality.csv"
            checkpoint = outdir / "quality" / f"{feature}_checkpoint.json"
            saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            evidence = saved.get("record", {}).get("audit_files_sha256", {})
            evidence_valid = "audit_files_sha256" in saved.get("record", {}) and all((outdir / p).is_file() and file_sha256(outdir / p) == digest for p, digest in evidence.items())
            if saved.get("config") == config and evidence_valid and all(p.exists() and saved.get("sha256", {}).get(p.name) == file_sha256(p) for p in (target, audit_path)):
                print(f"[RESUME] Enhanced {feature} checksum verified", flush=True)
                quality = pd.read_csv(audit_path, dtype={"point_status": str, "causality_mask": str})
                record = saved["record"]
            else:
                print(f"[ENHANCE] {feature}", flush=True)
                if feature in LEGACY_FEATURES:
                    oldpath = legacy_dir / entries[feature]["filename"]
                    frame = pd.read_csv(oldpath, float_precision="round_trip", low_memory=False)
                    original_values = frame[feature_columns(feature)].to_numpy(dtype=float)
                    quality = _read_quality(legacy_dir / "quality" / f"{feature}_quality.csv", original_values)
                    pd.testing.assert_frame_equal(metadata, frame[columns], check_exact=True, check_dtype=False)
                    if feature in ("P4", "P5", "P6"):
                        result, quality = _proton_overlay(feature, frame, quality, sgps_store, outdir / "continuation_audit")
                        result_values = result[feature_columns(feature)].to_numpy(dtype=float)
                        keep = np.isfinite(original_values) & (original_values != MISSING)
                        if not np.array_equal(original_values[keep], result_values[keep]):
                            raise AssertionError(f"Original measured/filled values changed: {feature}")
                        _atomic_csv(target, result)
                        del result, result_values
                    else:
                        _copy_atomic(oldpath, target)
                    del frame, original_values
                    source_metadata = sgps_store.metadata(feature) if feature in ("P4", "P5", "P6") else entries[feature].get("source_metadata", {})
                    sources = list(sgps_store.source_records) if feature in ("P4", "P5", "P6") else []
                else:
                    result, quality = _magnetic_feature(feature, metadata, magnetic_store)
                    _atomic_csv(target, result)
                    del result
                    source_metadata, sources = magnetic_store.metadata(feature), list(magnetic_store.source_records)
                restored = pd.read_csv(target, float_precision="round_trip", low_memory=False)
                pd.testing.assert_frame_equal(metadata, restored[columns], check_exact=True, check_dtype=False)
                if list(restored.columns) != columns + feature_columns(feature):
                    raise AssertionError(f"Wrong 340-column export layout: {feature}")
                del restored
                quality["sample_id"] = sample_audit.sample_id.to_numpy()
                _atomic_csv(audit_path, quality)
                record = {"feature": feature, "filename": name, "rows": len(metadata), "columns": 340,
                          "source_metadata": source_metadata, "source_records": sources}
                evidence_paths = list((outdir / "continuation_audit").glob(f"{feature}_*.csv"))
                if feature in ("P4", "P5", "P6"):
                    evidence_paths += list((outdir / "continuation_audit").glob("native_sgps_*.csv"))
                record["audit_files_sha256"] = {str(p.relative_to(outdir)): file_sha256(p) for p in evidence_paths}
                saved = {"config": config, "record": record,
                         "sha256": {p.name: file_sha256(p) for p in (target, audit_path)}}
                _atomic_json(checkpoint, saved)
            results.append({**record, "sha256": saved["sha256"][name], "quality_sha256": saved["sha256"][audit_path.name]})
            summaries.append(_summary(feature, quality))
            _atomic_json(state_path, {"state": "running", "completed_features": [r["feature"] for r in results], "config": config})
        for name in ("extension_samples.csv", "flare_cohort.csv", "label_audit.csv", "label_checkpoint.json"):
            if (legacy_dir / name).exists():
                _copy_atomic(legacy_dir / name, outdir / name)
        _atomic_csv(outdir / "quality_summary.csv", pd.DataFrame([{k: v for k, v in s.items() if k != "point_counts"} for s in summaries]))
        manifest = {"workflow": "enhanced_twelve_feature_research_release", "status": "complete_with_reported_unavailability",
                    "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                    "legacy_dir": str(legacy_dir.resolve()), "total_rows_per_feature": len(metadata),
                    "metadata_columns": columns, "feature_files": results, "quality_summary": summaries,
                    "proton_continuation": "D=log-energy/log-flux SGPS estimate at nominal GOES-15 reported energies; H=bounded causal hold. These are not identical EPEAD responses or cross-calibrated legacy fluxes.",
                    "magnetic_sampling": "native MDI/HMI observations; integration guard and bounded backward hold; R_VALUE is already a log index",
                    "legacy_preservation": "All original nonmissing values and all sample metadata/labels are unchanged; original ten files remain in legacy_dir."}
        _atomic_json(outdir / "dataset_manifest.json", manifest)
        (outdir / "availability_report.txt").write_text("\n".join([
            "Enhanced twelve-feature dataset", f"Rows per feature: {len(metadata)}; original 52 metadata + 288 values.",
            "Original ten-file release remains unchanged in: " + str(legacy_dir.resolve()),
            "P4/P5/P6 after March 4, 2020 use SGPS-derived flux at 30.6, 63.1 and 165 MeV where available.",
            "D denotes an energy-interpolated estimate, H denotes a bounded causal hold. This is not an identical detector response or a validated universal calibration.",
            "USFLUXL is line-of-sight unsigned active-region flux; R_VALUE is the native log R index. MDI/HMI calibration differs.",
            "Missing/ambiguous active-region matches, failed QC, and long source gaps remain -9999 and are counted below.",
            *[f"{s['feature']}: missing={s['missing_fraction_after']:.6f}; derived={s['derived_fraction']:.6f}; held={s['held_fraction']:.6f}" for s in summaries]
        ]) + "\n", encoding="utf-8")
        _atomic_json(state_path, {"state": manifest["status"], "completed_features": list(FEATURES), "config": config})
        return manifest
    except BaseException as exc:
        _atomic_json(state_path, {"state": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", "error": f"{type(exc).__name__}: {exc}", "config": config})
        raise


def validate_enhanced_dataset(dataset_dir: str | Path) -> dict:
    root = Path(dataset_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    if manifest.get("workflow") != "enhanced_twelve_feature_research_release":
        raise ValueError("Not an enhanced dataset manifest")
    if manifest["config"] != json.loads((root / "build_config.json").read_text()):
        raise ValueError("Enhanced manifest/config mismatch")
    if [e["feature"] for e in manifest["feature_files"]] != list(FEATURES):
        raise ValueError("Enhanced release must contain all twelve feature files")
    metadata, sample_ids = None, None
    for entry in manifest["feature_files"]:
        feature, path = entry["feature"], root / entry["filename"]
        qpath = root / "quality" / f"{feature}_quality.csv"
        if file_sha256(path) != entry["sha256"] or file_sha256(qpath) != entry["quality_sha256"]:
            raise ValueError(f"Enhanced checksum mismatch: {feature}")
        for relative, digest in entry.get("audit_files_sha256", {}).items():
            evidence = root / relative
            if not evidence.is_file() or file_sha256(evidence) != digest:
                raise ValueError(f"Enhanced source audit checksum mismatch: {relative}")
        frame = pd.read_csv(path, float_precision="round_trip", low_memory=False)
        cols = manifest["metadata_columns"]
        if list(frame.columns) != cols + feature_columns(feature) or len(frame) != manifest["total_rows_per_feature"]:
            raise ValueError(f"Enhanced schema/row count mismatch: {feature}")
        if metadata is None:
            metadata = frame[cols]
        else:
            pd.testing.assert_frame_equal(metadata, frame[cols], check_exact=True, check_dtype=False)
        if not frame.event_type.isin([0, 1]).all():
            raise ValueError(f"Invalid labels: {feature}")
        values = frame[feature_columns(feature)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite enhanced values: {feature}")
        q = pd.read_csv(qpath, dtype={"point_status": str, "causality_mask": str})
        if len(q) != len(frame) or q.row_index.tolist() != list(range(len(frame))):
            raise ValueError(f"Misaligned enhanced quality rows: {feature}")
        if sample_ids is None:
            sample_ids = q.sample_id.astype(str)
        elif not sample_ids.equals(q.sample_id.astype(str)):
            raise ValueError(f"Enhanced sample identifiers differ across features: {feature}")
        if not q.point_status.str.fullmatch("[OIEMDHSGU]{288}").all():
            raise ValueError(f"Invalid enhanced point mask: {feature}")
        masks = np.array([list(s) for s in q.point_status], dtype="U1")
        if not np.array_equal(np.isin(masks, ["S", "G"]), values == MISSING):
            raise ValueError(f"Enhanced missingness/mask mismatch: {feature}")
        if not np.array_equal((values == MISSING).sum(axis=1), q.missing_points_after):
            raise ValueError(f"Enhanced missing count mismatch: {feature}")
        print(f"[VALID] Enhanced {feature}: {len(frame)} rows", flush=True)
    return {"valid": True, "feature_files": len(FEATURES), "rows_per_file": len(metadata), "status": manifest["status"]}

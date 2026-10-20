"""Append observed flare samples to the ten frozen original feature files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .compatible_baseline import (FEATURES, STEPS, BASELINES, baseline_fingerprint,
                                  load_baseline, locate_baseline, read_original_metadata,
                                  window_columns)
from .compatible_catalogs import ASSOCIATION_RULE, METHOD_URL, fetch_goes_flares, label_flare_events
from .compatible_sources import CompatibleSourceStore

VERSION = "original-ten-strict-v3"
MISSING = -9999.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_csv(part, index=False)
    os.replace(part, path)


def window_index(onset: object) -> pd.DatetimeIndex:
    """Original 288 bins: ceil(flare, 5 min) - 24h ... - 5min."""
    stamp = pd.to_datetime(onset, utc=True).ceil("5min")
    return pd.date_range(end=stamp - pd.Timedelta(minutes=5), periods=STEPS, freq="5min")


def fill_window(raw: np.ndarray, structural: np.ndarray, medians: np.ndarray, *, return_status: bool = False):
    """Linear interior, nearest edge, historical median fallback; never fill structural absence."""
    values = np.asarray(raw, dtype=float).copy()
    structural, medians = np.asarray(structural, dtype=bool), np.asarray(medians, dtype=float)
    if values.shape != structural.shape or values.shape != medians.shape:
        raise ValueError("Raw, structural mask and historical medians must have matching shapes")
    values[(values == MISSING) | ~np.isfinite(values) | structural] = np.nan
    observed = np.isfinite(values)
    status = np.full(values.shape, "G", dtype="U1")
    status[observed], status[structural] = "O", "S"
    interpolated = edge = median = 0
    covered = np.flatnonzero(~structural)
    # Split at structural boundaries, so even edge filling cannot cross eras.
    for segment in np.split(covered, np.flatnonzero(np.diff(covered) != 1) + 1):
        if not len(segment):
            continue
        original = pd.Series(values[segment])
        inside = original.interpolate(method="linear", limit_area="inside")
        status[segment[(original.isna() & inside.notna()).to_numpy()]] = "I"
        interpolated += int((original.isna() & inside.notna()).sum())
        edged = inside.ffill().bfill()
        status[segment[(inside.isna() & edged.notna()).to_numpy()]] = "E"
        edge += int((inside.isna() & edged.notna()).sum())
        local = edged.to_numpy()
        needs = ~np.isfinite(local)
        usable = needs & np.isfinite(medians[segment])
        local[usable] = medians[segment][usable]
        status[segment[usable]] = "M"
        median += int(usable.sum())
        values[segment] = local
    counts = {"observed_points": int(observed.sum()), "interpolated_points": interpolated,
                    "edge_filled_points": edge, "median_filled_points": median,
                    "structural_missing_points": int(structural.sum()),
                    "missing_points_after": int((~np.isfinite(values)).sum()),
                    "unavailable_channel": bool(structural.any())}
    return (values, counts, "".join(status)) if return_status else (values, counts)


def _config(start: str, end: str) -> dict:
    for name, value in (("extension-start", start), ("end-date", end)):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"{name} must be YYYY-MM-DD")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first < pd.Timestamp("2014-01-01") or last < first:
        raise ValueError("Extension must start on/after 2014-01-01 and end on/after its start")
    if last > pd.Timestamp.now(tz="UTC").tz_localize(None).normalize():
        raise ValueError("Cannot extend with future observations")
    return {"version": VERSION, "historical_start": "1998-01-01", "historical_end": "2013-12-31",
            "extension_start": start, "end_date_inclusive": end, "channel_policy": "strict",
            "association_rule": ASSOCIATION_RULE, "features": list(FEATURES),
            "cadence_minutes": 5, "steps": STEPS}


def _load_extension(outdir: Path, cache: Path, metadata: pd.DataFrame,
                    store: CompatibleSourceStore, config: dict) -> tuple[pd.DataFrame, dict]:
    checkpoint = outdir / "label_checkpoint.json"
    files = [outdir / "extension_samples.csv", outdir / "label_audit.csv", outdir / "flare_cohort.csv"]
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("config") == config and all(p.exists() and saved.get("sha256", {}).get(p.name) == file_sha256(p) for p in files):
            print("[RESUME] Validated cached extension labels", flush=True)
            return pd.read_csv(files[0], float_precision="round_trip", low_memory=False), saved
    flares = fetch_goes_flares(config["extension_start"], config["end_date_inclusive"], cache / "flares")
    _atomic_csv(files[2], flares)
    labels, audit = label_flare_events(flares, flux_provider=store.integral_range)
    _atomic_csv(files[1], audit)
    if labels.empty:
        raise RuntimeError("No extension labels could be resolved from observations; inspect label_audit.csv")
    extension = labels.reindex(columns=metadata.columns).copy()
    if "Unnamed: 0" in metadata:
        first_id = int(pd.to_numeric(metadata["Unnamed: 0"], errors="raise").max()) + 1
        extension["Unnamed: 0"] = np.arange(first_id, first_id + len(extension))
    extension["event_type"] = pd.to_numeric(extension["event_type"], errors="raise").astype(int)
    # Match the historical text formats so existing user code can parse the
    # concatenated time columns without timezone/mixed-format surprises. All
    # stored timestamps are UTC; no historical strings are rewritten.
    for col in ("FlrOnset", "Flrmaxtime", "Flrendtime", "sep_start", "sep_end", "start_time", "peak_time", "end_time"):
        if col in extension:
            dates = pd.to_datetime(extension[col], utc=True, errors="coerce", format="mixed")
            fmt = "%Y-%m-%dT%H:%M:%S.000" if col in ("start_time", "peak_time", "end_time") else "%Y-%m-%d %H:%M:%S"
            extension[col] = dates.dt.strftime(fmt)
    _atomic_csv(files[0], extension)
    saved = {"config": config, "sha256": {p.name: file_sha256(p) for p in files},
             "sources": list(store.source_records), "resolved_rows": len(extension),
             "excluded_label_rows": int(audit.event_type.isna().sum()),
             "status_counts": audit.status.value_counts().to_dict()}
    _atomic_json(checkpoint, saved)
    return extension, saved


def _build_feature(feature: str, baseline: pd.DataFrame, extension: pd.DataFrame,
                   store: CompatibleSourceStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = window_columns(feature)
    original = baseline[columns].to_numpy(dtype=float)
    reference = np.where(np.isfinite(original) & (original != MISSING), original, np.nan)
    medians = pd.DataFrame(reference).median(axis=0).to_numpy(dtype=float)
    usable = reference[np.isfinite(reference)]
    medians[~np.isfinite(medians)] = float(np.median(usable)) if len(usable) else np.nan
    nold, nnew = len(baseline), len(extension)
    filled, rows = np.full((nnew, STEPS), np.nan), []
    for i in range(nold):
        rows.append({"row_index": i, "sample_id": f"historical-{i}", "origin": "baseline_frozen",
                     "observed_points": np.nan, "interpolated_points": np.nan,
                     "edge_filled_points": np.nan, "median_filled_points": np.nan,
                     "structural_missing_points": np.nan, "causality_excluded_points": np.nan,
                     "missing_points_after": int((~np.isfinite(reference[i])).sum()),
                     "unavailable_channel": False, "source_era": "published_1998_2013_filled",
                     "source_satellites": "published_baseline; individual_raw_source_not_reconstructed",
                     "point_status": "U" * STEPS, "causality_mask": "U" * STEPS,
                     "source_method": "unchanged_published_values; raw_prefill_quality_unknown"})
    onsets = pd.to_datetime(extension["FlrOnset"], utc=True)
    months = onsets.dt.strftime("%Y-%m")
    info = store.metadata(feature)
    lookahead = pd.Timedelta(minutes=info["bin_lookahead_minutes"])
    for month in months.unique():
        positions = np.flatnonzero(months.eq(month))
        first_grid = window_index(onsets.iloc[positions[0]])
        last_grid = window_index(onsets.iloc[positions[-1]])
        low, high = first_grid[0], last_grid[-1] + pd.Timedelta(minutes=5)
        print(f"[FEATURE] {feature} {month}: {len(positions)} flare windows", flush=True)
        series = store.range(feature, low, high)
        quality = store.quality_range(feature, low, high)
        for i in positions:
            grid = window_index(onsets.iloc[i])
            raw = series.reindex(grid).to_numpy(dtype=float)
            q = quality.reindex(grid)
            if q.channel_status.isna().any():
                raise RuntimeError(f"Incomplete source coverage metadata for {feature} at {onsets.iloc[i]}")
            structural = q.channel_status.eq("outside_original_channel_coverage").to_numpy()
            causal_excluded = (grid + lookahead > onsets.iloc[i]) & ~structural
            raw[causal_excluded] = np.nan
            filled[i], counts, point_status = fill_window(raw, structural, medians, return_status=True)
            counts.update({"row_index": nold + int(i), "sample_id": f"extension-{i}", "origin": "extension",
                           "causality_excluded_points": int(causal_excluded.sum()),
                           "point_status": point_status,
                           "causality_mask": "".join(np.where(causal_excluded, "1", "0")),
                           "source_era": " | ".join(sorted(q.source_product.astype(str).unique())),
                           "source_satellites": " | ".join(sorted(q.source_satellite.astype(str).unique())),
                           "source_method": "raw_native; causal_guard; linear_inside; nearest_edge; frozen_historical_median_fallback"})
            rows.append(counts)
    new = pd.concat([extension.reset_index(drop=True), pd.DataFrame(np.where(np.isfinite(filled), filled, MISSING), columns=columns)], axis=1)
    return pd.concat([baseline, new], ignore_index=True), pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


def _quality_summary(feature: str, quality: pd.DataFrame) -> dict:
    audited = quality.observed_points.notna()
    n = int(audited.sum())
    return {"feature": feature, "historical_rows": int((quality.origin == "baseline_frozen").sum()),
            "audited_extension_rows": n,
            "missing_fraction_before": float((STEPS - quality.loc[audited, "observed_points"]).sum() / (n * STEPS)) if n else None,
            "missing_fraction_after": float(quality.missing_points_after.sum() / (len(quality) * STEPS)),
            "whole_window_imputed_count": int((quality.median_filled_points == STEPS).sum()),
            "unavailable_window_count": int(quality.unavailable_channel.sum()),
            "structural_missing_points": int(quality.structural_missing_points.sum()),
            "causality_excluded_points": int(quality.causality_excluded_points.sum()),
            "prefill_quality_scope": "extension_only; historical_prefill_unknown"}


def _feature_sources(records: list[dict], feature: str) -> list[dict]:
    dataset = "OMNI_HRO2_5MIN" if feature in FEATURES[:5] else "goesp_xray_flux_P1M"
    if feature in ("P4", "P5", "P6"):
        return [r for r in records if r.get("product") == "p17ew" or r.get("source") == "structural_unavailability"]
    return [r for r in records if r.get("dataset") == dataset]


def build_compatible_dataset(*, end_date: str = "2025-12-31", extension_start: str = "2014-01-01",
                             outdir: str | Path = "data/processed/compatible_1998_2025",
                             cache_dir: str | Path = "data/cache/compatible",
                             baseline_dir: str | Path | None = None) -> dict:
    config = _config(extension_start, end_date)
    outdir, cache = Path(outdir), Path(cache_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "build_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory belongs to a different build; choose a separate --outdir")
    _atomic_json(config_path, config)
    status_path = outdir / "build_status.json"
    _atomic_json(status_path, {"state": "running", "config": config})
    store = CompatibleSourceStore(cache / "sources")
    try:
        metadata = read_original_metadata()
        extension, label_record = _load_extension(outdir, cache, metadata, store, config)
        results, summaries, unavailable = [], [], []
        for feature in FEATURES:
            print(f"[BUILD] {feature}: preserving {len(metadata)} historical samples", flush=True)
            path = locate_baseline(feature, cache, baseline_dir)
            baseline = load_baseline(feature, path, metadata)
            signature = file_sha256(path)
            name = f"1998_{pd.Timestamp(end_date).year}_MEMSEP_dataset_with_{feature}_filled.csv"
            target, quality_path = outdir / name, outdir / "quality" / f"{feature}_quality.csv"
            checkpoint = outdir / "quality" / f"{feature}_checkpoint.json"
            inputs = {"config": config, "baseline_sha256": signature,
                      "extension_sha256": label_record["sha256"]["extension_samples.csv"]}
            saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            if saved.get("inputs") == inputs and all(p.exists() and file_sha256(p) == saved.get("sha256", {}).get(p.name) for p in (target, quality_path)):
                print(f"[RESUME] Verified completed {feature}", flush=True)
                quality, record = pd.read_csv(quality_path), saved["record"]
            else:
                combined, quality = _build_feature(feature, baseline, extension, store)
                _atomic_csv(target, combined)
                del combined
                restored = pd.read_csv(target, float_precision="round_trip", low_memory=False)
                pd.testing.assert_frame_equal(baseline.reset_index(drop=True), restored.iloc[:len(baseline)].reset_index(drop=True), check_dtype=False, check_exact=True)
                if list(restored.columns) != list(metadata.columns) + window_columns(feature):
                    raise AssertionError(f"Export changed original column layout: {feature}")
                del restored
                _atomic_csv(quality_path, quality)
                record = {"feature": feature, "filename": name, "rows": len(baseline) + len(extension),
                          "columns": len(baseline.columns), "historical_rows": len(baseline),
                          "extension_rows": len(extension), "baseline_sha256": signature,
                          "historical_numeric_fingerprint": baseline_fingerprint(baseline, feature),
                          "historical_full_frame_roundtrip_verified": True,
                          "baseline_source": str(path) if baseline_dir else BASELINES[feature].url,
                          "source_metadata": store.metadata(feature), "source_records": _feature_sources(store.source_records, feature)}
                saved = {"inputs": inputs, "record": record,
                         "sha256": {p.name: file_sha256(p) for p in (target, quality_path)}}
                _atomic_json(checkpoint, saved)
            summaries.append(_quality_summary(feature, quality))
            results.append({**record, "sha256": saved["sha256"][name]})
            new_quality = quality[quality.origin.eq("extension")]
            for row in new_quality[new_quality.unavailable_channel].to_dict("records"):
                i = int(row["row_index"]) - len(metadata)
                unavailable.append({"feature": feature, "row_index": row["row_index"],
                                    "FlrOnset": str(extension.iloc[i].FlrOnset),
                                    "structural_missing_points": row["structural_missing_points"],
                                    "reason": "original_channel_unavailable; no_substitution_or_imputation"})
            del baseline, quality
            print(f"[DONE] {target}: {summaries[-1]['unavailable_window_count']} windows with reported channel absence", flush=True)
            _atomic_json(status_path, {"state": "running", "completed_features": [r["feature"] for r in results], "config": config})
        _atomic_csv(outdir / "quality_summary.csv", pd.DataFrame(summaries))
        _atomic_csv(outdir / "unavailable_channels.csv", pd.DataFrame(unavailable, columns=["feature", "row_index", "FlrOnset", "structural_missing_points", "reason"]))
        missing_metadata = [{"column": col, "extension_missing_rows": int(extension[col].isna().sum()),
                             "reason": "not_reported_or_legacy_convention_unreproducible"}
                            for col in metadata.columns if extension[col].isna().any()]
        _atomic_csv(outdir / "unavailable_metadata.csv", pd.DataFrame(missing_metadata, columns=["column", "extension_missing_rows", "reason"]))
        manifest = {"status": "complete_with_reported_unavailability" if unavailable or missing_metadata or label_record["excluded_label_rows"] else "complete",
                    "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                    "historical_rows": len(metadata), "extension_rows": len(extension),
                    "total_rows_per_feature": len(metadata) + len(extension),
                    "labels": {"method_url": METHOD_URL, "association_rule": ASSOCIATION_RULE,
                               "excluded_label_rows": label_record["excluded_label_rows"],
                               "status_counts": label_record["status_counts"], "source_records": label_record["sources"],
                               "limitations": "Historical labels unchanged. New competing-flare grouping is documented, not bitwise reproduction of unpublished code. Original sep_fluence normalization unknown; new sep_fluence fields missing; pfu-second fluences in label_audit.csv."},
                    "baseline_quality": "Historical filled values preserved; historical prefill missingness and imputation masks unknown.",
                    "feature_files": results, "quality_summary": summaries,
                    "partial_extension_range": extension_start != "2014-01-01" or end_date != "2025-12-31"}
        _atomic_json(outdir / "dataset_manifest.json", manifest)
        report = ["Original ten-feature extension", f"Range: historical 1998–2013 unchanged; requested extension {extension_start} through {end_date} inclusive.",
                  f"Samples: {len(metadata)} historical + {len(extension)} observed, labelled extension flares per file.",
                  f"Excluded unresolved labels: {label_record['excluded_label_rows']} (label_audit.csv).",
                  "P4/P5/P6 after 2020-03-04 23:55 UTC are unavailable. They remain -9999, including in _filled.csv files; no replacement bands or median fill.",
                  f"Unavailable channel windows: {len(unavailable)} across the ten files (unavailable_channels.csv).",
                  "Historical metadata and all 288 historical values passed exact numerical/string round-trip checks.",
                  "New gap filling: interior linear interpolation; nearest observed edges; frozen historical medians only for wholly missing covered segments. Quality masks record each operation.",
                  "Historical prefill quality cannot be reconstructed from published filled files.",
                  "Unreported original metadata stays missing; legacy fluence units are not guessed (unavailable_metadata.csv).",
                  "Native instrument responses/calibration eras differ; source_metadata in dataset_manifest.json records them.",
                  "A completed build does not assert that every original channel exists through 2025."]
        (outdir / "availability_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        _atomic_json(status_path, {"state": manifest["status"], "completed_features": list(FEATURES), "config": config})
        return manifest
    except BaseException as exc:
        state = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        _atomic_json(status_path, {"state": state, "error": f"{type(exc).__name__}: {exc}", "config": config})
        raise


def validate_dataset(dataset_dir: str | Path) -> dict:
    """Verify every final file, row/column alignment, checkpoints and checksums."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    metadata = read_original_metadata()
    if manifest["config"] != json.loads((root / "build_config.json").read_text()):
        raise ValueError("Manifest and build configuration disagree")
    labels = json.loads((root / "label_checkpoint.json").read_text())
    for filename, digest in labels["sha256"].items():
        if file_sha256(root / filename) != digest:
            raise ValueError(f"Label/cohort checksum mismatch: {filename}")
    extension = pd.read_csv(root / "extension_samples.csv", float_precision="round_trip", low_memory=False)
    if [r["feature"] for r in manifest["feature_files"]] != list(FEATURES):
        raise ValueError("Manifest must contain the ten original features in order")
    aligned = None
    for record in manifest["feature_files"]:
        feature, path = record["feature"], root / record["filename"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        frame = pd.read_csv(path, float_precision="round_trip", low_memory=False)
        if len(frame) != manifest["total_rows_per_feature"] or list(frame.columns) != list(metadata.columns) + window_columns(feature):
            raise ValueError(f"Wrong row count or column layout: {path}")
        current = frame[list(metadata.columns)]
        pd.testing.assert_frame_equal(extension.reset_index(drop=True), current.iloc[len(metadata):].reset_index(drop=True), check_dtype=False, check_exact=True)
        if aligned is None:
            aligned = current
        else:
            pd.testing.assert_frame_equal(aligned, current, check_dtype=False, check_exact=True)
        if not frame.event_type.isin([0, 1]).all():
            raise ValueError(f"Unknown SEP/NSEP labels: {path}")
        values = frame[window_columns(feature)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite feature data; use documented -9999 sentinel: {path}")
        quality = pd.read_csv(root / "quality" / f"{feature}_quality.csv", dtype={"point_status": str, "causality_mask": str})
        if len(quality) != len(frame) or quality.row_index.tolist() != list(range(len(frame))):
            raise ValueError(f"Quality rows do not align: {feature}")
        if not np.array_equal((values == MISSING).sum(axis=1), quality.missing_points_after.to_numpy()):
            raise ValueError(f"Missing values and quality records disagree: {feature}")
        if "point_status" not in quality or not quality.point_status.str.fullmatch(r"[OIEMSGU]{288}").all():
            raise ValueError(f"Invalid per-point quality mask: {feature}")
        new_quality = quality.iloc[len(metadata):]
        if new_quality.point_status.str.contains("U").any():
            raise ValueError(f"Unknown prefill quality in extension: {feature}")
        for code, count in (("O", "observed_points"), ("I", "interpolated_points"), ("E", "edge_filled_points"), ("M", "median_filled_points"), ("S", "structural_missing_points")):
            if not np.array_equal(new_quality.point_status.str.count(code).to_numpy(), new_quality[count].to_numpy()):
                raise ValueError(f"Quality mask and {count} disagree: {feature}")
        if not new_quality.causality_mask.str.fullmatch(r"[01]{288}").all():
            raise ValueError(f"Invalid causal mask: {feature}")
        if not np.array_equal(new_quality.causality_mask.str.count("1").to_numpy(), new_quality.causality_excluded_points.to_numpy()):
            raise ValueError(f"Causal mask and counts disagree: {feature}")
        masks = np.array([list(s) for s in new_quality.point_status], dtype="U1")
        if not np.array_equal(np.isin(masks, ["S", "G"]), values[len(metadata):] == MISSING):
            raise ValueError(f"Per-point availability and feature values disagree: {feature}")
        checkpoint = json.loads((root / "quality" / f"{feature}_checkpoint.json").read_text())
        if file_sha256(root / "quality" / f"{feature}_quality.csv") != checkpoint["sha256"][f"{feature}_quality.csv"]:
            raise ValueError(f"Quality checksum mismatch: {feature}")
        if baseline_fingerprint(frame.iloc[:len(metadata)], feature) != record["historical_numeric_fingerprint"]:
            raise ValueError(f"Historical data changed: {feature}")
        print(f"[VALID] {feature}: {len(frame)} rows, original {len(frame.columns)}-column layout", flush=True)
    return {"valid": True, "feature_files": len(FEATURES), "rows_per_file": manifest["total_rows_per_feature"], "status": manifest["status"]}

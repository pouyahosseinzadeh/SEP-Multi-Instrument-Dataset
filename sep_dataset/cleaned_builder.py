"""Create an aligned, gap-free release from selected collected feature files.

This is an offline derivative. Missing values are filled only from valid values
in the same 24-hour row. Samples with an entirely unavailable feature are
excluded from all selected files; no other sample supplies an invented row.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .compatible_builder import MISSING, STEPS, _atomic_csv, _atomic_json, file_sha256
from .enhanced_builder import FEATURES, feature_columns

VERSION = "aligned-rowwise-clean-v1"
WORKFLOW = "cleaned_twelve_feature_release"
TEN_FEATURES = FEATURES[:10]
TEN_WORKFLOW = "cleaned_ten_feature_release"
SOURCE_FILL_VALUES = {
    "F": 9999.99,
    "Vx": 99999.9,
    "flow_speed": 99999.9,
    "proton_density": 999.99,
    "T": 9999999.0,
}
STATUS_CODES = "OIEMDHSGU"


def sanitize_values(values, feature):
    """Return float values with missing codes as NaN and a source-fill mask.

    Positive OMNI fill values are matched exactly in their decimal and float32
    representations. Real zero values and signed velocities are not missing.
    The large-value guard also rejects conventional CDF-style fill magnitudes;
    none of these twelve physical quantities legitimately approaches 1e29.
    """
    data = np.asarray(values, dtype=float).copy()
    source_fill = np.isfinite(data) & (np.abs(data) >= 1e29)
    if feature in SOURCE_FILL_VALUES:
        fill = SOURCE_FILL_VALUES[feature]
        source_fill |= (data == fill) | (data == float(np.float32(fill)))
    data[(data == MISSING) | ~np.isfinite(data) | source_fill] = np.nan
    return data, source_fill


def interpolate_row(values, statuses):
    """Fill one chronological window; preserve every valid supplied value.

    Linear interpolation is used between valid anchors. Leading/trailing gaps
    receive the nearest available value (E), including a one-anchor row. Both
    methods are imputation, not independent native observations.
    """
    data = np.asarray(values, dtype=float).copy()
    status = np.asarray(list(statuses) if isinstance(statuses, str) else statuses, dtype="U1").copy()
    if data.ndim != 1 or len(data) != STEPS or status.shape != data.shape:
        raise ValueError("A cleaning row must have 288 feature values and status codes")
    valid = np.isfinite(data)
    if not valid.any():
        raise ValueError("An entirely unavailable feature row has no interpolation anchors")
    missing = ~valid
    anchors = np.flatnonzero(valid)
    if missing.any():
        positions = np.flatnonzero(missing)
        data[missing] = np.interp(positions, anchors, data[valid])
        interior = missing & (np.arange(STEPS) > anchors[0]) & (np.arange(STEPS) < anchors[-1])
        status[interior] = "I"
        status[missing & ~interior] = "E"
    return data, "".join(status)


def _load_quality(path, rows):
    q = pd.read_csv(path, dtype={"point_status": str, "causality_mask": str, "sample_id": str,
                                 "source_point_status": str, "before_point_status": str,
                                 "cleaning_fill_mask": str}, low_memory=False)
    if not {"row_index", "sample_id", "point_status"}.issubset(q):
        raise ValueError(f"Quality sidecar lacks row_index, sample_id or point_status: {path}")
    if len(q) != rows or q.row_index.tolist() != list(range(rows)):
        raise ValueError(f"Quality rows are not aligned: {path}")
    if q.sample_id.isna().any() or q.sample_id.duplicated().any():
        raise ValueError(f"Quality sample identifiers must be present and unique: {path}")
    if not q.point_status.str.fullmatch(f"[{STATUS_CODES}]{{{STEPS}}}").all():
        raise ValueError(f"Invalid point-status mask: {path}")
    return q


def _read_feature(root, entry, metadata_columns, expected_rows):
    filename = entry["filename"]
    if Path(filename).name != filename:
        raise ValueError("Feature filenames must be direct children of the dataset folder")
    frame = pd.read_csv(root / filename, float_precision="round_trip", low_memory=False)
    columns = feature_columns(entry["feature"])
    if list(frame.columns) != metadata_columns + columns or len(frame) != expected_rows:
        raise ValueError(f"Wrong input feature schema or row count: {filename}")
    if not frame.event_type.isin([0, 1]).all():
        raise ValueError(f"Invalid SEP/NSEP labels: {filename}")
    return frame


def _before_status(statuses, invalid):
    masks = np.array([list(s) for s in statuses], dtype="U1")
    masks[invalid & ~np.isin(masks, ["S", "G"])] = "G"
    return masks


def _class_counts(metadata):
    return {"NSEP": int(metadata.event_type.eq(0).sum()), "SEP": int(metadata.event_type.eq(1).sum())}


def _cohort_counts(metadata, keep):
    years = pd.to_datetime(metadata.FlrOnset, utc=True, errors="raise", format="mixed").dt.year
    table = pd.DataFrame({"year": years, "event_type": metadata.event_type, "retained": keep.astype(int)})
    result = table.groupby(["year", "event_type"], sort=True).retained.agg(["size", "sum"]).reset_index()
    result = result.rename(columns={"size": "source_samples", "sum": "retained_samples"})
    result["excluded_samples"] = result.source_samples - result.retained_samples
    return result


def _mask_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def _run_length(mask):
    positions = np.flatnonzero(mask)
    if not len(positions):
        return 0
    return max(len(block) for block in np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1))


def _clean_quality(q, values, source_fill, keep):
    """Preserve source audit fields and add exact cleaning provenance."""
    invalid = ~np.isfinite(values)
    before_masks = _before_status(q.point_status, invalid)
    q = q.loc[keep].copy().reset_index(drop=True)
    original_rows = np.flatnonzero(keep)
    q["source_row_index"] = original_rows
    q["row_index"] = np.arange(len(q))
    q["source_point_status"] = q.point_status.astype(str)
    q["before_point_status"] = ["".join(m) for m in before_masks[keep]]
    q["cleaning_fill_mask"] = ["".join(np.where(row, "1", "0")) for row in invalid[keep]]
    q["cleaning_input_valid_points"] = (~invalid[keep]).sum(axis=1)
    q["cleaning_source_fill_points"] = source_fill[keep].sum(axis=1)
    q["source_fill_points"] = q.cleaning_source_fill_points
    q["cleaning_missing_points_before"] = invalid[keep].sum(axis=1)
    q["cleaning_longest_missing_run_points"] = [_run_length(row) for row in invalid[keep]]
    q["source_unavailable_channel"] = q.get("unavailable_channel", q.cleaning_missing_points_before.gt(0))
    selected = values[keep].copy()
    cleaned_masks = []
    linear_counts, edge_counts = [], []
    for i, row in enumerate(selected):
        data, mask = interpolate_row(row, before_masks[original_rows[i]])
        selected[i] = data
        cleaned_masks.append(mask)
        newly_filled = invalid[original_rows[i]]
        letters = np.array(list(mask))
        linear_counts.append(int((newly_filled & (letters == "I")).sum()))
        edge_counts.append(int((newly_filled & (letters == "E")).sum()))
    q["point_status"] = cleaned_masks
    q["cleaning_interpolated_points"] = linear_counts
    q["cleaning_edge_filled_points"] = edge_counts
    q["missing_points_after"] = 0
    q["unavailable_channel"] = False
    for code, field in (("O", "observed_points"), ("I", "interpolated_points"), ("E", "edge_filled_points"),
                        ("M", "median_filled_points"), ("D", "derived_points"), ("H", "held_points"),
                        ("S", "structural_missing_points")):
        q[field] = q.point_status.str.count(code)
    unknown = q.point_status.str.contains("U")
    # A preserved filled historical value is still of unknown original origin.
    # Do not make the legacy "before interpolation" report treat it as observed.
    for field in ("observed_points", "interpolated_points", "edge_filled_points", "median_filled_points"):
        q.loc[unknown, field] = np.nan
    methods = q.get("source_method", pd.Series("unknown", index=q.index)).fillna("unknown").astype(str)
    q["source_method"] = methods + "; analysis_cleaning=rowwise_linear_interior_and_nearest_edge"
    return selected, q


def _quality_summary(feature, q):
    points = len(q) * STEPS
    counts = {code: int(q.point_status.str.count(code).sum()) for code in STATUS_CODES}
    return {"feature": feature, "samples": len(q), "missing_fraction_after": 0.0,
            "derived_fraction": counts["D"] / points, "held_fraction": counts["H"] / points,
            "unknown_quality_fraction": counts["U"] / points,
            "whole_window_imputed_count": int(q.point_status.eq("M" * STEPS).sum()),
            "unavailable_windows": 0, "cleaning_interpolated_points": int(q.cleaning_interpolated_points.sum()),
            "cleaning_edge_filled_points": int(q.cleaning_edge_filled_points.sum()), "point_counts": counts}


def _selected_features(features):
    selected = tuple(FEATURES if features is None else features)
    if selected not in (tuple(FEATURES), tuple(TEN_FEATURES)):
        raise ValueError("Cleaning supports the original ten features or all twelve features in their original order")
    return selected


def _check_output_features(output, entries):
    """Refuse mixed releases without deleting or overwriting unrelated files."""
    expected_files = {entry["filename"] for entry in entries}
    extra_files = {p.name for p in output.glob("*_with_*_filled.csv")} - expected_files
    expected_quality = {f"{entry['feature']}_quality.csv" for entry in entries}
    extra_quality = {p.name for p in (output / "quality").glob("*_quality.csv")} - expected_quality
    if extra_files or extra_quality:
        raise ValueError("Cleaned output contains other feature files; choose a new --outdir for the selected feature cohort")


def clean_dataset(dataset_dir="data/processed/enhanced_1998_2025", *, outdir="data/processed/cleaned_1998_2025", features=None):
    """Create ten or twelve aligned feature files from the source archive, offline.

    The ten-feature cohort is selected before cleaning, so unavailable magnetic
    rows cannot exclude usable samples from the original ten feature files.
    """
    root, output = Path(dataset_dir), Path(outdir)
    features = _selected_features(features)
    workflow = TEN_WORKFLOW if features == tuple(TEN_FEATURES) else WORKFLOW
    feature_label = "ten" if workflow == TEN_WORKFLOW else "twelve"
    if root.resolve() == output.resolve():
        raise ValueError("Cleaned output needs its own folder; collected inputs must remain unchanged")
    source_manifest = root / "dataset_manifest.json"
    if not source_manifest.is_file():
        raise ValueError(f"No completed source manifest in {root}; finish enhancement before cleaning")
    original = json.loads(source_manifest.read_text())
    if original.get("workflow") in (WORKFLOW, TEN_WORKFLOW):
        raise ValueError("Cleaning requires the unfiltered enhanced source archive, not an already cleaned dataset; excluded rows cannot be recovered from cleaned files")
    source_entries = original.get("feature_files", [])
    if workflow == TEN_WORKFLOW:
        if original.get("workflow") != "enhanced_twelve_feature_research_release":
            raise ValueError("Ten-feature cleanup requires the unfiltered enhanced source archive")
        by_feature = {entry.get("feature"): entry for entry in source_entries}
        if len(by_feature) != len(source_entries) or any(feature not in by_feature for feature in features):
            raise ValueError("Source manifest must contain each original ten feature exactly once")
        entries = [by_feature[feature] for feature in features]
    else:
        entries = source_entries
        if [e.get("feature") for e in entries] != list(features):
            raise ValueError("Cleaning requires a complete twelve-feature dataset in the original feature order")
    _check_output_features(output, entries)
    output_manifest = output / "dataset_manifest.json"
    if output_manifest.is_file():
        saved_features = [entry.get("feature") for entry in json.loads(output_manifest.read_text()).get("feature_files", [])]
        if saved_features != list(features):
            raise ValueError("Cleaned folder belongs to a different feature cohort; choose a new --outdir")
    columns = original.get("metadata_columns", [])
    if len(columns) != 52 or not {"FlrOnset", "event_type"}.issubset(columns):
        raise ValueError("Cleaning requires the original 52 metadata columns")
    rows = original.get("total_rows_per_feature")
    if type(rows) is not int or rows <= 0:
        raise ValueError("Source manifest has no positive sample count")

    metadata, ids = None, None
    empty_by_feature, source_sizes, quality_sizes = {}, {}, {}
    invalid_counts, fill_counts, input_hashes, quality_hashes = {}, {}, {}, {}
    for entry in entries:
        feature = entry["feature"]
        print(f"[CLEAN SCAN] {feature}: checking {rows} samples", flush=True)
        path, qpath = root / entry["filename"], root / "quality" / f"{feature}_quality.csv"
        digest, qdigest = file_sha256(path), file_sha256(qpath)
        if entry.get("sha256") and digest != entry["sha256"]:
            raise ValueError(f"Source checksum mismatch: {feature}")
        if entry.get("quality_sha256") and qdigest != entry["quality_sha256"]:
            raise ValueError(f"Source quality checksum mismatch: {feature}")
        frame = _read_feature(root, entry, columns, rows)
        q = _load_quality(qpath, rows)
        if metadata is None:
            metadata, ids = frame[columns].copy(), q.sample_id.copy()
        else:
            pd.testing.assert_frame_equal(metadata, frame[columns], check_dtype=False, check_exact=True)
            if not ids.equals(q.sample_id):
                raise ValueError(f"Input sample identifiers differ across files: {feature}")
        values, source_fill = sanitize_values(frame[feature_columns(feature)].to_numpy(dtype=float), feature)
        # Quality-coded unavailable points cannot become valid merely because a
        # malformed upstream file gave them a finite numeric placeholder.
        masks = np.array([list(s) for s in q.point_status], dtype="U1")
        values[np.isin(masks, ["S", "G"])] = np.nan
        empty_by_feature[feature] = ~np.isfinite(values).any(axis=1)
        invalid_counts[feature] = int((~np.isfinite(values)).sum())
        fill_counts[feature] = int(source_fill.sum())
        source_sizes[feature], quality_sizes[feature] = path.stat().st_size, qpath.stat().st_size
        input_hashes[feature], quality_hashes[feature] = digest, qdigest
        del frame, q, values, source_fill, masks

    excluded = np.logical_or.reduce(list(empty_by_feature.values()))
    keep = ~excluded
    retained = int(keep.sum())
    source_manifest_hash = file_sha256(source_manifest)
    source_config_hash = hashlib.sha256(json.dumps(original.get("config", {}), sort_keys=True, default=str).encode()).hexdigest()
    config = {"version": VERSION, "features": list(features), "source_config_sha256": source_config_hash,
              "input_files": input_hashes, "input_quality_files": quality_hashes,
              "cohort_policy": f"exclude a sample from all {feature_label} files when any selected feature has no valid anchors" if workflow == TEN_WORKFLOW else "exclude a sample from all twelve files when any feature has no valid anchors",
              "fill_policy": "linear interpolation within a row; nearest valid edge; one-anchor rows held constant; no cross-row fill",
              "source_fill_values": SOURCE_FILL_VALUES, "source_fill_matching": "exact decimal or float32 representation; abs>=1e29 also rejected"}
    output.mkdir(parents=True, exist_ok=True)
    config_path, state_path = output / "build_config.json", output / "build_status.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cleaned folder belongs to different inputs or policy; choose a new --outdir")
    if (output / "dataset_manifest.json").is_file():
        try:
            result = validate_cleaned_dataset(output)
            print(f"[RESUME] All {feature_label} cleaned files verified: {result['rows_per_file']} samples each", flush=True)
            return json.loads((output / "dataset_manifest.json").read_text())
        except (ValueError, OSError, AssertionError, KeyError):
            pass  # Repair inconsistent outputs from verified source files below.
    _atomic_json(config_path, config)
    _atomic_json(state_path, {"state": "running", "config": config})
    try:
        mapping = pd.DataFrame({"source_row_index": range(rows), "sample_id": ids,
                                "FlrOnset": metadata.FlrOnset, "event_type": metadata.event_type})
        exclusions = mapping.loc[excluded].copy()
        exclusions["empty_features"] = [" | ".join(f for f in features if empty_by_feature[f][i]) for i in np.flatnonzero(excluded)]
        for feature in features:
            exclusions[f"empty_{feature}"] = empty_by_feature[feature][excluded]
        _atomic_csv(output / "exclusions.csv", exclusions)
        retained_map = mapping.loc[keep].copy().reset_index(drop=True)
        retained_map.insert(0, "row_index", range(retained))
        _atomic_csv(output / "row_mapping.csv", retained_map)
        cohort = _cohort_counts(metadata, keep)
        _atomic_csv(output / "cohort_counts.csv", cohort)
        report = {"source_samples": rows, "retained_samples": retained, "excluded_samples": int(excluded.sum()),
                  "class_counts_before": _class_counts(metadata), "class_counts_after": _class_counts(metadata.loc[keep]),
                  "empty_feature_rows": {f: int(empty_by_feature[f].sum()) for f in features},
                  "missing_points_before_all_source_rows": invalid_counts, "source_fill_points_all_source_rows": fill_counts,
                  "policy": config["fill_policy"], "global_alignment": config["cohort_policy"],
                  "limitations": "Interpolation and edge filling are labelled imputations within retrospective pre-flare windows, not native measurements; long gaps may be filled and are explicitly counted. No entirely missing row is synthesized."}
        _atomic_json(output / "cleaning_report.json", report)
        if not retained:
            raise ValueError("No common samples remain: every row has an entirely unavailable feature. See exclusions.csv and cleaning_report.json; no measurements were fabricated.")

        # Estimate the remaining CSV/audit allocation, including an atomic-write
        # temporary file. This is a lower-risk preflight, not a storage guarantee.
        ratio = retained / rows
        estimate = sum(source_sizes.values()) * ratio * 1.25 + sum(quality_sizes.values()) * ratio * 4.0
        estimate += max(source_sizes.values()) * ratio * 1.25
        already = sum(p.stat().st_size for p in output.glob("*.csv")) + sum(p.stat().st_size for p in (output / "quality").glob("*.csv"))
        needed = max(0, int(estimate - already))
        free = shutil.disk_usage(output).free
        print(f"[CLEAN COHORT] {retained}/{rows} retained; {int(excluded.sum())} globally excluded", flush=True)
        print(f"[CLEAN STORAGE] estimated additional {needed / 1024**3:.2f} GiB; available {free / 1024**3:.2f} GiB", flush=True)
        if free < needed:
            raise RuntimeError(f"Insufficient free storage for cleaned outputs: estimated additional {needed / 1024**3:.2f} GiB, available {free / 1024**3:.2f} GiB. Keep inputs/caches and free space or choose --outdir on another disk.")
        report["storage_estimate_additional_bytes"] = needed
        _atomic_json(output / "cleaning_report.json", report)
        results, summaries, cleaning_rows = [], [], []
        kept_metadata = metadata.loc[keep].reset_index(drop=True)
        for entry in entries:
            feature = entry["feature"]
            target = output / entry["filename"]
            qtarget = output / "quality" / f"{feature}_quality.csv"
            checkpoint = output / "quality" / f"{feature}_checkpoint.json"
            saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            if saved.get("config") == config and all(p.exists() and saved.get("sha256", {}).get(p.name) == file_sha256(p) for p in (target, qtarget)):
                print(f"[RESUME] Cleaned {feature} checksum verified", flush=True)
                q = pd.read_csv(qtarget, dtype={"point_status": str, "sample_id": str})
                record = saved["record"]
            else:
                print(f"[CLEAN] {feature}: interpolate within {retained} retained rows", flush=True)
                frame = _read_feature(root, entry, columns, rows)
                q = _load_quality(root / "quality" / f"{feature}_quality.csv", rows)
                raw, source_fill = sanitize_values(frame[feature_columns(feature)].to_numpy(dtype=float), feature)
                source_masks = np.array([list(s) for s in q.point_status], dtype="U1")
                raw[np.isin(source_masks, ["S", "G"])] = np.nan
                anchor_values = raw[keep][np.isfinite(raw[keep])]
                anchor_hash = _mask_hash(anchor_values)
                data, q = _clean_quality(q, raw, source_fill, keep)
                if not np.isfinite(data).all() or (data == MISSING).any():
                    raise AssertionError(f"Cleaning left unavailable values: {feature}")
                anchor_mask = np.isfinite(raw[keep])
                if not np.array_equal(data[anchor_mask], anchor_values):
                    raise AssertionError(f"Cleaning changed an existing valid value: {feature}")
                result = pd.concat([kept_metadata, pd.DataFrame(data, columns=feature_columns(feature))], axis=1)
                _atomic_csv(target, result)
                _atomic_csv(qtarget, q)
                restored = pd.read_csv(target, float_precision="round_trip", low_memory=False)
                pd.testing.assert_frame_equal(kept_metadata, restored[columns], check_dtype=False, check_exact=True)
                record = {"feature": feature, "filename": entry["filename"], "rows": retained, "columns": 340,
                          "source_metadata": entry.get("source_metadata", {}),
                          "source_file_sha256": input_hashes[feature], "source_quality_sha256": quality_hashes[feature],
                          "input_valid_values_sha256": anchor_hash,
                          "cleaning": "Row-wise linear interior interpolation; nearest edge; global removal of wholly unavailable samples"}
                saved = {"config": config, "record": record,
                         "sha256": {p.name: file_sha256(p) for p in (target, qtarget)}}
                _atomic_json(checkpoint, saved)
                del frame, raw, source_fill, source_masks, data, result, restored, anchor_values, anchor_mask
            results.append({**record, "sha256": saved["sha256"][target.name], "quality_sha256": saved["sha256"][qtarget.name]})
            summary = _quality_summary(feature, q)
            summaries.append(summary)
            cleaning_rows.append({"feature": feature, "source_samples": rows, "retained_samples": retained,
                                  "empty_feature_rows_before": int(empty_by_feature[feature].sum()),
                                  "missing_points_before_all_source_rows": invalid_counts[feature],
                                  "source_fill_points_all_source_rows": fill_counts[feature],
                                  "missing_points_before_retained": int(q.cleaning_missing_points_before.sum()),
                                  "source_fill_points_retained": int(q.cleaning_source_fill_points.sum()),
                                  "new_linear_points": int(q.cleaning_interpolated_points.sum()),
                                  "new_edge_points": int(q.cleaning_edge_filled_points.sum()),
                                  "one_anchor_rows": int(q.cleaning_input_valid_points.eq(1).sum()), "missing_points_after": 0})
            _atomic_json(state_path, {"state": "running", "completed_features": [r["feature"] for r in results], "config": config})
            del q
        _atomic_csv(output / "quality_summary.csv", pd.DataFrame([{k: v for k, v in s.items() if k != "point_counts"} for s in summaries]))
        _atomic_csv(output / "feature_cleaning_summary.csv", pd.DataFrame(cleaning_rows))
        availability_text = "\n".join([
            f"Cleaned {feature_label}-feature analysis dataset",
            f"Source samples: {rows}; retained: {retained}; globally excluded: {int(excluded.sum())}.",
            f"Retained labels: SEP={report['class_counts_after']['SEP']}; NSEP={report['class_counts_after']['NSEP']}.",
            f"Every retained row has 288 finite feature values in each of the {feature_label} aligned files; no -9999 feature values remain.",
            "Interior gaps use linear interpolation (I); leading/trailing gaps use nearest available edges (E). One-anchor rows use the single available value and are explicitly flagged.",
            "No row is filled from another sample or its label. A sample with a wholly unavailable feature is removed from every feature file; see exclusions.csv.",
            "Existing O/D/H/U provenance is retained for unchanged values. Filled values are imputation, not new native observations; historical prefill provenance may remain unknown (U).",
            "Known OMNI positive fill codes are treated as unavailable before interpolation; see feature_cleaning_summary.csv.",
            "The original collected datasets, source caches and native audit files remain unchanged in the source dataset folder.",
            *[f"{r['feature']}: retained={retained}; missing_after=0; new_linear_points={r['new_linear_points']}; new_edge_points={r['new_edge_points']}; wholly_unavailable_source_rows={r['empty_feature_rows_before']}" for r in cleaning_rows]
        ]) + "\n"
        availability_path = output / "availability_report.txt"
        availability_part = availability_path.with_suffix(".txt.part")
        availability_part.write_text(availability_text, encoding="utf-8")
        availability_part.replace(availability_path)
        artifacts = {name: file_sha256(output / name) for name in ("exclusions.csv", "row_mapping.csv", "cohort_counts.csv", "cleaning_report.json", "quality_summary.csv", "feature_cleaning_summary.csv", "availability_report.txt")}
        manifest = {"workflow": workflow, "status": "complete_gap_free_analysis_derivative",
                    "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                    "source_dataset_dir": str(root.resolve()), "source_manifest_sha256": source_manifest_hash,
                    "source_total_rows": rows, "excluded_rows": int(excluded.sum()), "total_rows_per_feature": retained,
                    "metadata_columns": columns, "feature_files": results, "quality_summary": summaries,
                    "audit_files_sha256": artifacts,
                    "imputation_note": report["limitations"], "cleaning_policy": config["fill_policy"],
                    "class_counts_before": report["class_counts_before"],
                    "class_counts_after": report["class_counts_after"],
                    "source_audits": "Original native/source measurement audits remain in source_dataset_dir; they are not duplicated here."}
        _atomic_json(output / "dataset_manifest.json", manifest)
        _atomic_json(state_path, {"state": "complete", "completed_features": list(features), "config": config})
        validate_cleaned_dataset(output)
        print(f"[DONE] {feature_label.capitalize()} cleaned files: {retained} aligned samples each; no unavailable feature values", flush=True)
        return manifest
    except BaseException as exc:
        try:
            _atomic_json(state_path, {"state": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                                      "error": f"{type(exc).__name__}: {exc}", "config": config})
        except OSError:
            pass  # Retain the original storage error rather than masking it.
        raise


def validate_cleaned_dataset(dataset_dir):
    """Verify the gap-free cohort, metadata, provenance, values and audit files."""
    root = Path(dataset_dir)
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Cleaned dataset is incomplete in {root}; run or resume clean first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("workflow") not in (WORKFLOW, TEN_WORKFLOW):
        raise ValueError("Not a cleaned ten- or twelve-feature dataset manifest")
    features = tuple(TEN_FEATURES) if manifest["workflow"] == TEN_WORKFLOW else tuple(FEATURES)
    if manifest.get("config", {}).get("features") != list(features):
        raise ValueError("Cleaned workflow/config feature selection disagrees")
    if manifest["config"] != json.loads((root / "build_config.json").read_text()):
        raise ValueError("Cleaned manifest/config mismatch")
    if json.loads((root / "build_status.json").read_text()).get("state") != "complete":
        raise ValueError("Cleaned build is not complete; resume clean before analysis")
    if [entry["feature"] for entry in manifest["feature_files"]] != list(features):
        raise ValueError("Cleaned output requires exactly its selected features in their original order")
    _check_output_features(root, manifest["feature_files"])
    rows = manifest["total_rows_per_feature"]
    if rows <= 0:
        raise ValueError("Cleaned dataset cannot be empty")
    if manifest["source_total_rows"] != rows + manifest["excluded_rows"]:
        raise ValueError("Cleaned source/retained/excluded counts disagree")
    for name, digest in manifest["audit_files_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Cleaned audit checksum mismatch: {name}")
    mapping = pd.read_csv(root / "row_mapping.csv", dtype={"sample_id": str})
    exclusions = pd.read_csv(root / "exclusions.csv", dtype={"sample_id": str})
    if len(mapping) != rows or len(exclusions) != manifest["excluded_rows"] or mapping.row_index.tolist() != list(range(rows)):
        raise ValueError("Cleaned row mapping/exclusions do not reconcile")
    if mapping.sample_id.duplicated().any() or exclusions.sample_id.duplicated().any() or set(mapping.sample_id) & set(exclusions.sample_id):
        raise ValueError("Cleaned retained/excluded sample identifiers overlap or duplicate")
    if not mapping.event_type.isin([0, 1]).all() or not exclusions.event_type.isin([0, 1]).all():
        raise ValueError("Invalid retained/excluded labels")
    empty_columns = [f"empty_{feature}" for feature in features]
    if not set(empty_columns).issubset(exclusions) or (len(exclusions) and not exclusions[empty_columns].any(axis=1).all()):
        raise ValueError("Excluded samples lack an entirely unavailable feature reason")
    original_indices = mapping.source_row_index.tolist() + exclusions.source_row_index.tolist()
    if sorted(original_indices) != list(range(manifest["source_total_rows"])):
        raise ValueError("Cleaned row mapping does not partition the original samples")
    metadata = None
    for entry in manifest["feature_files"]:
        feature = entry["feature"]
        path, qpath = root / entry["filename"], root / "quality" / f"{feature}_quality.csv"
        if file_sha256(path) != entry["sha256"] or file_sha256(qpath) != entry["quality_sha256"]:
            raise ValueError(f"Cleaned checksum mismatch: {feature}")
        frame = _read_feature(root, entry, manifest["metadata_columns"], rows)
        q = _load_quality(qpath, rows)
        current_metadata = frame[manifest["metadata_columns"]]
        if metadata is None:
            metadata = current_metadata.copy()
        else:
            pd.testing.assert_frame_equal(metadata, current_metadata, check_dtype=False, check_exact=True)
        if not q.sample_id.equals(mapping.sample_id) or not q.source_row_index.equals(mapping.source_row_index):
            raise ValueError(f"Cleaned identifiers/source rows differ from the retained cohort: {feature}")
        values = frame[feature_columns(feature)].to_numpy(dtype=float)
        sanitized, _ = sanitize_values(values, feature)
        if not np.isfinite(sanitized).all():
            raise ValueError(f"Unavailable values or documented source fill codes remain: {feature}")
        if not q.point_status.str.fullmatch(f"[OIEMDHU]{{{STEPS}}}").all() or not q.missing_points_after.eq(0).all():
            raise ValueError(f"Invalid cleaned status mask or missing counts: {feature}")
        if not q.cleaning_fill_mask.astype(str).str.fullmatch(f"[01]{{{STEPS}}}").all():
            raise ValueError(f"Invalid cleaning-fill mask: {feature}")
        if not q.source_point_status.str.fullmatch(f"[{STATUS_CODES}]{{{STEPS}}}").all() or not q.before_point_status.str.fullmatch(f"[{STATUS_CODES}]{{{STEPS}}}").all():
            raise ValueError(f"Invalid source/before-cleaning status mask: {feature}")
        # Read the zero-prefixed binary mask as text, not a lossy integer.
        mask_text = q.cleaning_fill_mask
        newly_filled = np.array([list(s) for s in mask_text], dtype="U1") == "1"
        final_masks = np.array([list(s) for s in q.point_status], dtype="U1")
        source_masks = np.array([list(s) for s in q.source_point_status], dtype="U1")
        before_masks = np.array([list(s) for s in q.before_point_status], dtype="U1")
        if not np.array_equal(final_masks[~newly_filled], source_masks[~newly_filled]):
            raise ValueError(f"Existing valid source provenance changed: {feature}")
        if not np.array_equal(before_masks[~newly_filled], source_masks[~newly_filled]) or not np.isin(before_masks[newly_filled], ["S", "G"]).all():
            raise ValueError(f"Before-cleaning source availability disagrees: {feature}")
        if not np.isin(final_masks[newly_filled], ["I", "E"]).all():
            raise ValueError(f"Newly filled points lack interpolation/edge provenance: {feature}")
        if not np.array_equal(newly_filled.sum(axis=1), q.cleaning_missing_points_before.to_numpy()):
            raise ValueError(f"Cleaned missing-before/fill counts disagree: {feature}")
        if not np.array_equal((~newly_filled).sum(axis=1), q.cleaning_input_valid_points.to_numpy()):
            raise ValueError(f"Cleaned source anchor counts disagree: {feature}")
        if _mask_hash(values[~newly_filled]) != entry["input_valid_values_sha256"]:
            raise ValueError(f"Existing valid source values changed: {feature}")
        if not np.array_equal((newly_filled & (final_masks == "I")).sum(axis=1), q.cleaning_interpolated_points.to_numpy()):
            raise ValueError(f"Interpolation counts disagree: {feature}")
        if not np.array_equal((newly_filled & (final_masks == "E")).sum(axis=1), q.cleaning_edge_filled_points.to_numpy()):
            raise ValueError(f"Edge-fill counts disagree: {feature}")
        for code, field in (("O", "observed_points"), ("I", "interpolated_points"), ("E", "edge_filled_points"),
                            ("M", "median_filled_points"), ("D", "derived_points"), ("H", "held_points"),
                            ("S", "structural_missing_points")):
            expected = (final_masks == code).sum(axis=1)
            if code in "OIEM":
                known = ~(final_masks == "U").any(axis=1)
                if not q.loc[~known, field].isna().all() or not np.array_equal(q.loc[known, field].to_numpy(), expected[known]):
                    raise ValueError(f"Cleaned known/unknown {field} counts disagree: {feature}")
            elif not np.array_equal(q[field].to_numpy(), expected):
                raise ValueError(f"Cleaned {field} counts disagree: {feature}")
        print(f"[VALID] Cleaned {feature}: {rows} aligned rows; zero unavailable feature values", flush=True)
    if not metadata.event_type.reset_index(drop=True).equals(mapping.event_type.reset_index(drop=True)):
        raise ValueError("Cleaned labels differ from row mapping")
    if not metadata.FlrOnset.astype(str).reset_index(drop=True).equals(mapping.FlrOnset.astype(str).reset_index(drop=True)):
        raise ValueError("Cleaned flare onsets differ from row mapping")
    return {"valid": True, "feature_files": len(features), "rows_per_file": rows,
            "source_rows": manifest["source_total_rows"], "excluded_rows": manifest["excluded_rows"],
            "status": manifest["status"]}

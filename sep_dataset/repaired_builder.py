"""Repair provenance, proton consistency and physical-flare identities offline.

The enhanced archive is immutable input. A source adapter restores historical
raw missingness and supplies only independently validated proton estimates.
Feature windows are read one at a time; no duplicate staging release is made.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import cleaned_builder as cb
from .compatible_builder import _atomic_csv, _atomic_json, file_sha256
from .enhanced_builder import feature_columns
from .flare_integrity import IDENTITY_VERSION, audit_flare_identities, assert_flare_label_integrity


WORKFLOW = "repaired_ten_feature_release_v2"
FEATURES = tuple(cb.TEN_FEATURES)
VERSION = "raw-provenance-and-validated-protons-v2"


def _json_list(value, field):
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid JSON in " + field) from error
    if not isinstance(data, list) or any(not isinstance(item, str) for item in data):
        raise ValueError(field + " must contain a JSON list of strings")
    return data


def _native_key(value):
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid calibration source key") from error
    from .proton_calibration import KEY_FIELDS
    if not isinstance(data, dict) or set(data) != set(KEY_FIELDS) or any(not str(v).strip() for v in data.values()):
        raise ValueError("Calibration source identity is incomplete")
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _adapter_frame(adapter, entry, columns, rows, metadata, ids):
    frame, quality = adapter.read_feature(entry, columns, rows)
    feature = entry["feature"]
    if list(frame.columns) != columns + feature_columns(feature) or len(frame) != rows:
        raise ValueError("Repair adapter changed feature schema or row count: " + feature)
    pd.testing.assert_frame_equal(metadata, frame[columns], check_dtype=False, check_exact=True)
    needed = {"row_index", "sample_id", "point_status"}
    if not needed.issubset(quality) or len(quality) != rows or quality.row_index.tolist() != list(range(rows)):
        raise ValueError("Repair adapter quality rows are not aligned: " + feature)
    if not ids.equals(quality.sample_id.astype(str).reset_index(drop=True)):
        raise ValueError("Repair adapter changed sample identities: " + feature)
    if not quality.point_status.str.fullmatch("[OIEMDHSGU]{288}").all():
        raise ValueError("Invalid repaired source point mask: " + feature)
    values, source_fill = cb.sanitize_values(frame[feature_columns(feature)].to_numpy(dtype=float), feature)
    masks = np.array([list(str(v)) for v in quality.point_status], dtype="U1")
    values[np.isin(masks, ["S", "G"])] = np.nan
    # A whole row of median fallbacks is not an observed interpolation anchor.
    values[(masks == "M").all(axis=1)] = np.nan
    return frame, quality, values, source_fill


def _report_payload(adapter, name):
    reports = getattr(adapter, "reports", {})
    value = reports.get(name)
    if not isinstance(value, dict):
        raise ValueError("Repair adapter did not supply " + name)
    return value


def repair_dataset(dataset_dir="data/processed/enhanced_1998_2025", *,
                   outdir="data/processed/repaired_ten_1998_2025", calibration_bank=None,
                   historical_masks=None, source_adapter=None):
    """Write the aligned ten-feature derivative after all source/identity gates."""
    root, output = Path(dataset_dir), Path(outdir)
    if root.resolve() == output.resolve():
        raise ValueError("Repair output must be separate from the unfiltered enhanced archive")
    manifest_path = root / "dataset_manifest.json"
    original = json.loads(manifest_path.read_text())
    if original.get("workflow") != "enhanced_twelve_feature_research_release":
        raise ValueError("Repair requires the unfiltered enhanced archive, not an already filtered cleaned release")
    source_entries = original.get("feature_files", [])
    by_feature = {entry.get("feature"): entry for entry in source_entries}
    if len(by_feature) != len(source_entries) or any(f not in by_feature for f in FEATURES):
        raise ValueError("Enhanced manifest lacks the unique original ten features")
    entries = [by_feature[f] for f in FEATURES]
    columns, rows = original.get("metadata_columns", []), original.get("total_rows_per_feature")
    if len(columns) != 52 or not {"FlrOnset", "event_type"}.issubset(columns) or type(rows) is not int or rows <= 0:
        raise ValueError("Repair requires 52 original metadata columns and a positive source row count")
    cb._check_output_features(output, entries)
    old_manifest = output / "dataset_manifest.json"
    if old_manifest.exists() and json.loads(old_manifest.read_text()).get("workflow") != WORKFLOW:
        raise ValueError("Output belongs to an older release; use a new repair --outdir")
    input_hashes, quality_hashes, source_sizes, quality_sizes = {}, {}, {}, {}
    # Verify every supplied source before allowing the adapter to process it.
    for entry in entries:
        feature, filename = entry["feature"], entry["filename"]
        if Path(filename).name != filename:
            raise ValueError("Source feature filenames must be direct children of the archive")
        path, qpath = root / filename, root / "quality" / (feature + "_quality.csv")
        digest, qdigest = file_sha256(path), file_sha256(qpath)
        if digest != entry.get("sha256") or qdigest != entry.get("quality_sha256"):
            raise ValueError("Enhanced source checksum mismatch: " + feature)
        input_hashes[feature], quality_hashes[feature] = digest, qdigest
        source_sizes[feature], quality_sizes[feature] = path.stat().st_size, qpath.stat().st_size
    metadata = pd.read_csv(root / entries[0]["filename"], usecols=columns, float_precision="round_trip", low_memory=False)
    if len(metadata) != rows:
        raise ValueError("Source metadata row count disagrees with manifest")
    first_quality = cb._load_quality(root / "quality" / (FEATURES[0] + "_quality.csv"), rows)
    ids = first_quality.sample_id.astype(str).reset_index(drop=True)
    identity = audit_flare_identities(metadata)
    canonical = identity["canonical_ids"].reset_index(drop=True)
    if source_adapter is None:
        from .repair_sources import RepairAdapter
        source_adapter = RepairAdapter(root, original, calibration_bank=calibration_bank, historical_masks=historical_masks)
    empty_by_feature, invalid_counts, fill_counts, adapted_hashes = {}, {}, {}, {}
    for entry in entries:
        feature = entry["feature"]
        print("[REPAIR SCAN] {}: {} source rows".format(feature, rows), flush=True)
        frame, q, values, source_fill = _adapter_frame(source_adapter, entry, columns, rows, metadata, ids)
        empty_by_feature[feature] = ~np.isfinite(values).any(axis=1)
        invalid_counts[feature], fill_counts[feature] = int((~np.isfinite(values)).sum()), int(source_fill.sum())
        adapted_hashes[feature] = {"valid_values": cb._mask_hash(values[np.isfinite(values)]),
                                   "point_status": hashlib.sha256(q.point_status.str.cat().encode()).hexdigest()}
        del frame, q, values, source_fill
    empty = np.logical_or.reduce(list(empty_by_feature.values()))
    flare_keep = np.asarray(identity["keep"], dtype=bool)
    keep, retained = flare_keep & ~empty, int((flare_keep & ~empty).sum())
    config = {"version": VERSION, "features": list(FEATURES), "source_manifest_sha256": file_sha256(manifest_path),
              "input_files": input_hashes, "input_quality_files": quality_hashes,
              "source_adapter": json.loads(json.dumps(source_adapter.config, sort_keys=True)),
              "adapted_source_fingerprints": adapted_hashes,
              "identity_policy": IDENTITY_VERSION, "cohort_policy": "Original ten only; remove wholly unavailable rows, whole-row median fallbacks, conflicting flare identities and duplicate physical flares globally",
              "fill_policy": "Linear within-row interpolation; nearest valid edges; no cross-row or label-based filling"}
    output.mkdir(parents=True, exist_ok=True)
    config_path, state_path = output / "build_config.json", output / "build_status.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Repair output has different source/calibration/policy inputs; choose a new --outdir")
    if old_manifest.exists():
        try:
            validate_repaired_dataset(output)
            print("[RESUME] Completed repaired release verified", flush=True)
            return json.loads(old_manifest.read_text())
        except (ValueError, OSError, AssertionError, KeyError):
            pass
    _atomic_json(config_path, config)
    _atomic_json(state_path, {"state": "running", "config": config})
    try:
        mapping = pd.DataFrame({"source_row_index": range(rows), "sample_id": ids,
                                "canonical_flare_id": canonical, "FlrOnset": metadata.FlrOnset,
                                "event_type": metadata.event_type})
        excluded = mapping.loc[~keep].copy()
        source_reasons = identity["audit"].set_index("source_row_position").exclusion_reason
        excluded["flare_identity_excluded"] = ~flare_keep[~keep]
        excluded["empty_features"] = [" | ".join(f for f in FEATURES if empty_by_feature[f][i]) for i in np.flatnonzero(~keep)]
        excluded["exclusion_reason"] = ["; ".join(x for x in ("flare_identity:" + str(source_reasons.loc[i]) if not flare_keep[i] else "", "wholly_unavailable:" + "|".join(f for f in FEATURES if empty_by_feature[f][i]) if empty[i] else "") if x) for i in np.flatnonzero(~keep)]
        for feature in FEATURES:
            excluded["empty_" + feature] = empty_by_feature[feature][~keep]
        _atomic_csv(output / "exclusions.csv", excluded)
        retained_map = mapping.loc[keep].reset_index(drop=True)
        retained_map.insert(0, "row_index", range(retained))
        _atomic_csv(output / "row_mapping.csv", retained_map)
        _atomic_csv(output / "cohort_counts.csv", cb._cohort_counts(metadata, keep))
        _atomic_csv(output / "flare_identity_audit.csv", identity["audit"])
        _atomic_json(output / "flare_identity_summary.json", identity["summary"])
        if not retained:
            raise ValueError("No samples remain after raw availability and physical-flare identity checks; see exclusions.csv")
        report = _report_payload(source_adapter, "proton_calibration_report")
        if report.get("status") != "passed":
            raise ValueError("Proton calibration/source policy did not pass; repaired release is blocked")
        historical_report = _report_payload(source_adapter, "historical_mask_report")
        _atomic_json(output / "proton_calibration_report.json", report)
        _atomic_json(output / "historical_mask_report.json", historical_report)
        audit_names = ["exclusions.csv", "row_mapping.csv", "cohort_counts.csv", "flare_identity_audit.csv", "flare_identity_summary.json", "proton_calibration_report.json", "historical_mask_report.json"]
        if hasattr(source_adapter, "write_audits"):
            for name in source_adapter.write_audits(output):
                name = str(name)
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or not (output / path).is_file():
                    raise ValueError("Adapter audit path must identify a saved file inside the repair output")
                audit_names.append(name)
        # Source audits are written before feature output; preflight accounts
        # for the remaining feature/quality allocation and an atomic temp file.
        ratio = retained / rows
        estimated = (sum(source_sizes.values()) + max(source_sizes.values())) * ratio * 1.25 + sum(quality_sizes.values()) * ratio * 4
        already = sum(p.stat().st_size for p in output.glob("*_filled.csv")) + sum(p.stat().st_size for p in (output / "quality").glob("*_quality.csv"))
        needed = max(0, int(estimated - already))
        free = shutil.disk_usage(output).free
        print("[REPAIR COHORT] {}/{} retained; {} excluded".format(retained, rows, rows-retained), flush=True)
        print("[REPAIR STORAGE] estimated additional {:.2f} GiB; available {:.2f} GiB".format(needed/1024**3, free/1024**3), flush=True)
        if free < needed:
            raise RuntimeError("Insufficient free storage for repaired outputs: need approximately {:.2f} additional GiB; {:.2f} GiB available. Free space or select --outdir on another disk.".format(needed/1024**3, free/1024**3))
        kept_metadata, results, summaries, cleaning_rows = metadata.loc[keep].reset_index(drop=True), [], [], []
        for entry in entries:
            feature = entry["feature"]
            target, qtarget = output / entry["filename"], output / "quality" / (feature + "_quality.csv")
            checkpoint = output / "quality" / (feature + "_checkpoint.json")
            saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            if saved.get("config") == config and all(p.is_file() and file_sha256(p) == saved.get("sha256", {}).get(p.name) for p in (target, qtarget)):
                q = cb._load_quality(qtarget, retained)
                record = saved["record"]
                print("[RESUME] Repaired " + feature, flush=True)
            else:
                print("[REPAIR] " + feature, flush=True)
                frame, q, raw, source_fill = _adapter_frame(source_adapter, entry, columns, rows, metadata, ids)
                if not np.array_equal(~np.isfinite(raw).any(axis=1), empty_by_feature[feature]):
                    raise ValueError("Repair adapter availability changed between scan and write: " + feature)
                fingerprint = {"valid_values": cb._mask_hash(raw[np.isfinite(raw)]),
                               "point_status": hashlib.sha256(q.point_status.str.cat().encode()).hexdigest()}
                if fingerprint != adapted_hashes[feature]:
                    raise ValueError("Repair adapter values/provenance changed between scan and write: " + feature)
                q = q.copy()
                q["canonical_flare_id"] = canonical.to_numpy()
                anchor = raw[keep][np.isfinite(raw[keep])]
                anchor_hash = cb._mask_hash(anchor)
                data, q = cb._clean_quality(q, raw, source_fill, keep)
                if not np.isfinite(data).all() or not np.array_equal(data[np.isfinite(raw[keep])], anchor):
                    raise AssertionError("Repair changed an adapted valid anchor or left a missing cell: " + feature)
                result = pd.concat([kept_metadata, pd.DataFrame(data, columns=feature_columns(feature))], axis=1)
                _atomic_csv(target, result)
                _atomic_csv(qtarget, q)
                record = {"feature": feature, "filename": entry["filename"], "rows": retained, "columns": 340,
                          "source_metadata": entry.get("source_metadata", {}), "source_file_sha256": input_hashes[feature],
                          "source_quality_sha256": quality_hashes[feature], "input_valid_values_sha256": anchor_hash,
                          "processing": "Raw-availability restoration and validated source adaptation, followed by row-wise interpolation and common identity/availability exclusion"}
                saved = {"config": config, "record": record, "sha256": {p.name: file_sha256(p) for p in (target, qtarget)}}
                _atomic_json(checkpoint, saved)
                del frame, raw, source_fill, anchor, data, result
            results.append({**record, "sha256": saved["sha256"][target.name], "quality_sha256": saved["sha256"][qtarget.name]})
            summaries.append(cb._quality_summary(feature, q))
            cleaning_rows.append({"feature": feature, "source_samples": rows, "retained_samples": retained,
                                  "empty_feature_rows_before": int(empty_by_feature[feature].sum()),
                                  "missing_points_before_all_source_rows": invalid_counts[feature],
                                  "source_fill_points_all_source_rows": fill_counts[feature],
                                  "missing_points_before_retained": int(q.cleaning_missing_points_before.sum()),
                                  "new_linear_points": int(q.cleaning_interpolated_points.sum()),
                                  "new_edge_points": int(q.cleaning_edge_filled_points.sum()),
                                  "missing_points_after": 0})
            _atomic_json(state_path, {"state": "running", "completed_features": [r["feature"] for r in results], "config": config})
        # The adapter must be deterministic and must not silently change the
        # calibration/missingness policy during the second feature pass.
        if json.loads(json.dumps(source_adapter.config, sort_keys=True)) != config["source_adapter"]:
            raise ValueError("Repair adapter configuration changed while writing")
        if hasattr(source_adapter, "verify_sources"):
            source_adapter.verify_sources()
        for entry in entries:
            feature = entry["feature"]
            if file_sha256(root / entry["filename"]) != input_hashes[feature] or file_sha256(root / "quality" / (feature + "_quality.csv")) != quality_hashes[feature]:
                raise ValueError("An archive input changed during repair: " + feature)
        if file_sha256(manifest_path) != config["source_manifest_sha256"]:
            raise ValueError("The source manifest changed during repair")
        _atomic_csv(output / "quality_summary.csv", pd.DataFrame([{k: v for k, v in s.items() if k != "point_counts"} for s in summaries]))
        _atomic_csv(output / "feature_cleaning_summary.csv", pd.DataFrame(cleaning_rows))
        cleaning_report = {"source_samples": rows, "retained_samples": retained, "excluded_samples": rows-retained,
                           "class_counts_before": cb._class_counts(metadata), "class_counts_after": cb._class_counts(kept_metadata),
                           "empty_feature_rows": {f: int(empty_by_feature[f].sum()) for f in FEATURES},
                           "identity_summary": identity["summary"], "policy": config["fill_policy"],
                           "limitations": "Filled cells are imputations; supported modern proton values are validated legacy-equivalent estimates within their recorded calibration domain, not native legacy measurements."}
        _atomic_json(output / "cleaning_report.json", cleaning_report)
        text = ["Repaired ten-feature analysis release", "Source samples: {}; retained: {}; excluded: {}.".format(rows, retained, rows-retained),
                "Retained SEP: {}; NSEP: {}.".format(cleaning_report["class_counts_after"]["SEP"], cleaning_report["class_counts_after"]["NSEP"]),
                "Every saved feature row has 288 finite values. Partial gaps are interpolated within the same row; entirely unavailable rows and identity conflicts are excluded globally.",
                "Historical raw missingness is restored before selection. Modern proton values require a passed source-specific calibration; unsupported source rows are excluded.",
                "The same physical flare cannot occur in both SEP and NSEP. See flare_identity_audit.csv and exclusions.csv.",
                cleaning_report["limitations"]]
        (output / "availability_report.txt").write_text("\n".join(text) + "\n")
        audit_names += ["quality_summary.csv", "feature_cleaning_summary.csv", "cleaning_report.json", "availability_report.txt"]
        manifest = {"workflow": WORKFLOW, "status": "pending_validation", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config, "source_dataset_dir": str(root.resolve()), "source_manifest_sha256": file_sha256(manifest_path),
                    "source_total_rows": rows, "excluded_rows": rows-retained, "total_rows_per_feature": retained,
                    "metadata_columns": columns, "feature_files": results, "quality_summary": summaries,
                    "audit_files_sha256": {name: file_sha256(output / name) for name in sorted(set(audit_names))},
                    "class_counts_before": cleaning_report["class_counts_before"], "class_counts_after": cleaning_report["class_counts_after"],
                    "cleaning_policy": config["fill_policy"], "identity_summary": identity["summary"],
                    "source_audits": "Native/source audit files remain in the immutable enhanced archive; calibration and raw-mask provenance are included in this derivative."}
        _atomic_json(output / "dataset_manifest.json", manifest)
        _atomic_json(state_path, {"state": "verifying", "completed_features": list(FEATURES), "config": config})
        cb.validate_cleaned_dataset(output, allow_verifying=True)
        manifest["status"] = "complete_repaired_gap_free_derivative"
        _atomic_json(output / "dataset_manifest.json", manifest)
        _atomic_json(state_path, {"state": "complete", "completed_features": list(FEATURES), "config": config})
        print("[DONE] Ten repaired files: {} aligned samples each".format(retained), flush=True)
        return manifest
    except BaseException as error:
        try:
            _atomic_json(state_path, {"state": "failed", "error": "{}: {}".format(type(error).__name__, error), "config": config})
        except OSError:
            pass
        raise


def validate_repaired_audits(root, manifest, mapping, exclusions, metadata):
    """Extra repaired-release checks; called after ordinary value/mask checks."""
    root = Path(root)
    required = {"flare_identity_audit.csv", "flare_identity_summary.json", "proton_calibration_report.json", "historical_mask_report.json", "calibration_bank.json"}
    if not required.issubset(manifest.get("audit_files_sha256", {})):
        raise ValueError("Repaired release is missing required identity, calibration or historical audits")
    audit = pd.read_csv(root / "flare_identity_audit.csv", dtype={"canonical_flare_id": str})
    if audit.source_row_position.tolist() != list(range(manifest["source_total_rows"])):
        raise ValueError("Flare identity audit does not cover the original source rows")
    if not audit.retained.isin([True, False]).all():
        raise ValueError("Flare identity selection must contain explicit booleans")
    selected = audit.iloc[mapping.source_row_index.to_numpy(dtype=int)]
    if not selected.retained.all() or not np.array_equal(selected.canonical_flare_id.to_numpy(), mapping.canonical_flare_id.to_numpy()):
        raise ValueError("Retained rows disagree with the physical-flare identity audit")
    if mapping.canonical_flare_id.isna().any() or mapping.canonical_flare_id.duplicated().any():
        raise ValueError("Retained canonical flare IDs must be unique")
    excluded_audit = audit.iloc[exclusions.source_row_index.to_numpy(dtype=int)]
    if not exclusions.flare_identity_excluded.isin([True, False]).all() or not np.array_equal(exclusions.flare_identity_excluded.to_numpy(dtype=bool), ~excluded_audit.retained.to_numpy(dtype=bool)):
        raise ValueError("Excluded identity reasons disagree with the source audit")
    if not np.array_equal(audit.event_type.iloc[mapping.source_row_index].to_numpy(), metadata.event_type.to_numpy()):
        raise ValueError("Identity audit labels changed in the retained dataset")
    assert_flare_label_integrity(metadata)
    report = json.loads((root / "proton_calibration_report.json").read_text())
    if report.get("status") != "passed":
        raise ValueError("Repaired proton calibration policy did not pass")
    for field in ("processed_native_keys", "validated_native_keys"):
        if not isinstance(report.get(field), list):
            raise ValueError("Calibration report must list " + field)
    processed = {_native_key(v) for v in report["processed_native_keys"]}
    validated = {_native_key(v) for v in report["validated_native_keys"]}
    if not processed.issubset(validated) or type(report.get("unsupported_rows_excluded")) is not int or not 0 <= report["unsupported_rows_excluded"] <= manifest["excluded_rows"]:
        raise ValueError("Unsupported or unvalidated proton sources entered the repaired release")
    bank = json.loads((root / "calibration_bank.json").read_text())
    if not isinstance(bank.get("artifacts"), list):
        raise ValueError("Calibration bank lacks an artifact list")
    from .proton_calibration import _check_artifact
    bank_ids, bank_keys = set(), set()
    for artifact in bank["artifacts"]:
        _check_artifact(artifact, require_pass=True)
        bank_ids.add(artifact["calibration_id"])
        key = artifact["applicability"]["source_key"].copy()
        for basis in artifact["applicability"].get("allowed_energy_bases", [key["energy_coordinate_basis"]]):
            key["energy_coordinate_basis"] = basis
            bank_keys.add(_native_key(key))
    if not validated.issubset(bank_keys):
        raise ValueError("Calibration report claims source identities not covered by passed artifacts")
    for feature in FEATURES:
        q = pd.read_csv(root / "quality" / (feature + "_quality.csv"), dtype={"canonical_flare_id": str, "point_status": str})
        if "canonical_flare_id" not in q or not q.canonical_flare_id.equals(mapping.canonical_flare_id):
            raise ValueError("Quality canonical identities disagree: " + feature)
        historical = pd.to_datetime(metadata.FlrOnset, utc=True, format="mixed").lt(pd.Timestamp("2014-01-01", tz="UTC"))
        if historical.any():
            if "historical_raw_missing_points" not in q:
                raise ValueError("Historical raw-availability provenance is missing: " + feature)
            counts = pd.to_numeric(q.loc[historical.to_numpy(), "historical_raw_missing_points"], errors="coerce")
            if counts.isna().any() or not counts.between(0, 287).all():
                raise ValueError("Wholly missing historical rows or unknown raw availability remain: " + feature)
        if feature in ("P4", "P5", "P6"):
            for field in ("source_calibration_keys", "calibration_ids"):
                if field not in q:
                    raise ValueError("Missing proton calibration provenance: " + field)
            for row in q.itertuples():
                keys = {_native_key(k) for k in _json_list(row.source_calibration_keys, "source_calibration_keys")}
                calibration_ids = set(_json_list(row.calibration_ids, "calibration_ids"))
                if not keys.issubset(processed) or not calibration_ids.issubset(bank_ids):
                    raise ValueError("A retained proton value uses an unsupported source/model")
                if ("D" in row.point_status or "H" in row.point_status) and (not keys or not calibration_ids):
                    raise ValueError("Derived proton values lack validated model provenance")


def validate_repaired_dataset(dataset_dir):
    root = Path(dataset_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    if manifest.get("workflow") != WORKFLOW:
        raise ValueError("Not a repaired ten-feature release")
    return cb.validate_cleaned_dataset(root)

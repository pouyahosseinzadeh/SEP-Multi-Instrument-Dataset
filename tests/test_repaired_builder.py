"""End-to-end cohort, identity, source-gate and preservation checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sep_dataset import cleaned_builder as cb, repaired_builder as rb
from test_cleaned_builder import fixture_enhanced, read_feature, read_quality, digest, columns
from test_proton_calibration import fitted


def _snapshot(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def artifact():
    _, _, model = fitted()
    assert model["status"] == "passed"
    return model


class Adapter:
    """A deterministic independently supplied source-policy fixture."""
    def __init__(self, root, model, *, mask_historical_p6=False, median_row=False, status="passed"):
        self.root, self.model = root, model
        self.mask_historical_p6, self.median_row = mask_historical_p6, median_row
        self.key = json.dumps(model["applicability"]["source_key"], sort_keys=True)
        self.config = {"fixture": "validated-adapter", "calibration_id": model["calibration_id"],
                       "raw_historical_p6_mask": mask_historical_p6, "whole_median_row": median_row}
        self.reports = {"historical_mask_report": {"status": "passed", "fixture": "known original raw availability"},
                        "proton_calibration_report": {"status": status, "encountered_native_keys": [self.key],
                                                      "processed_native_keys": [self.key], "validated_native_keys": [self.key],
                                                      "unsupported_rows_excluded": 0}}
        self.calls = 0

    def read_feature(self, entry, metadata_columns, rows):
        self.calls += 1
        feature = entry["feature"]
        frame = cb._read_feature(self.root, entry, metadata_columns, rows)
        q = cb._load_quality(self.root / "quality" / (feature + "_quality.csv"), rows)
        dates = pd.to_datetime(frame.FlrOnset, utc=True, format="mixed")
        historical = dates.lt(pd.Timestamp("2014-01-01", tz="UTC"))
        q["historical_raw_missing_points"] = np.where(historical, 0, np.nan)
        # Existing fixture has D/H examples before the modern source era.
        # Treat its finite pre-2020 values as its declared native measurements.
        modern = dates.ge(pd.Timestamp("2020-01-01", tz="UTC"))
        q.loc[~modern, "point_status"] = q.loc[~modern, "point_status"].str.replace("D", "O", regex=False).str.replace("H", "O", regex=False)
        q["source_calibration_keys"], q["calibration_ids"] = "[]", "[]"
        if feature in ("P4", "P5", "P6"):
            q.loc[modern, "point_status"] = q.loc[modern, "point_status"].str.replace("O", "D", regex=False)
            q.loc[modern, "source_calibration_keys"] = json.dumps([self.key])
            q.loc[modern, "calibration_ids"] = json.dumps([self.model["calibration_id"]])
        if self.mask_historical_p6 and feature == "P6":
            frame.loc[historical, columns(feature)] = -99999.0
            q.loc[historical, "point_status"] = "G" * 288
            q.loc[historical, "historical_raw_missing_points"] = 288
        if self.median_row and feature == "Vx":
            q.loc[1, "point_status"] = "M" * 288
        return frame, q

    def write_audits(self, outdir):
        path = outdir / "calibration_bank.json"
        path.write_text(json.dumps({"schema": "sep-proton-calibration-bank-v1", "artifacts": [self.model]}))
        return [path.name]


def _edit_source_metadata(root, changes):
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["feature_files"]:
        path = root / entry["filename"]
        frame = pd.read_csv(path, float_precision="round_trip")
        for row, values in changes.items():
            for name, value in values.items():
                frame.loc[row, name] = value
        frame.to_csv(path, index=False)
        entry["sha256"] = digest(path)
    manifest_path.write_text(json.dumps(manifest))


def test_repair_restores_raw_absence_and_excludes_whole_median_without_source_writes(tmp_path, artifact):
    source, output = tmp_path / "enhanced", tmp_path / "repaired"
    metadata = fixture_enhanced(source)
    before = _snapshot(source)
    adapter = Adapter(source, artifact, mask_historical_p6=True, median_row=True)
    result = rb.repair_dataset(source, outdir=output, source_adapter=adapter)
    assert result["workflow"] == rb.WORKFLOW
    assert result["total_rows_per_feature"] == 2
    assert len(result["feature_files"]) == 10
    assert adapter.calls == 20
    for feature in rb.FEATURES:
        frame, q = read_feature(output, feature), read_quality(output, feature)
        pd.testing.assert_frame_equal(frame.iloc[:, :52], metadata.iloc[[2, 3]].reset_index(drop=True), check_dtype=False)
        assert frame.shape == (2, 340)
        assert np.isfinite(frame[columns(feature)].to_numpy()).all()
        assert q.sample_id.tolist() == ["flare-2", "flare-3"]
        assert q.canonical_flare_id.nunique() == 2
        assert q.missing_points_after.eq(0).all()
    # The two valid F anchors survive; source fill-coded cells are replaced.
    final_f = read_feature(output, "F")
    assert final_f.loc[1, columns("F")[10]] == 5
    assert final_f.loc[1, columns("F")[280]] == 15
    exclusions = pd.read_csv(output / "exclusions.csv")
    assert exclusions.source_row_index.tolist() == [0, 1]
    assert exclusions.empty_P6.iloc[0] and exclusions.empty_Vx.iloc[1]
    assert rb.validate_repaired_dataset(output)["valid"]
    assert before == _snapshot(source)
    before_output = _snapshot(output)
    rb.repair_dataset(source, outdir=output, source_adapter=adapter)
    assert before_output == _snapshot(output)


def test_conflicting_sep_nsep_flare_is_quarantined_in_all_ten_files(tmp_path, artifact):
    source, output = tmp_path / "enhanced", tmp_path / "repaired"
    fixture_enhanced(source)
    _edit_source_metadata(source, {2: {"FlrOnset": "2014-01-02 03:00:00", "noaa_active_region": 11944}})
    rb.repair_dataset(source, outdir=output, source_adapter=Adapter(source, artifact))
    mapping = pd.read_csv(output / "row_mapping.csv")
    assert mapping.source_row_index.tolist() == [0, 3]
    exclusions = pd.read_csv(output / "exclusions.csv")
    assert exclusions.source_row_index.tolist() == [1, 2]
    assert exclusions.flare_identity_excluded.all()
    assert not exclusions[["empty_" + f for f in rb.FEATURES]].any(axis=None)
    assert exclusions.exclusion_reason.str.contains("conflicting_sep_nsep_labels").all()
    assert rb.validate_repaired_dataset(output)["feature_files"] == 10


def test_fresh_cohort_excludes_long_or_excessive_gaps_and_preserves_threshold_anchors(tmp_path, artifact):
    source, output = tmp_path / "native", tmp_path / "repaired"
    fixture_enhanced(source)
    manifest_path = source / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["workflow"] = rb.NATIVE_WORKFLOW
    manifest["feature_files"] = [entry for entry in manifest["feature_files"] if entry["feature"] in rb.FEATURES]
    manifest_path.write_text(json.dumps(manifest))

    class FreshAdapter(Adapter):
        def read_feature(self, entry, metadata_columns, rows):
            frame, quality = super().read_feature(entry, metadata_columns, rows)
            feature = entry["feature"]
            values = np.tile(np.arange(1., 289.), (rows, 1))
            if feature == "Vx":
                values *= -1
            masks = np.full(values.shape, "O", dtype="U1")
            if feature in ("P4", "P5", "P6"):
                masks[2:] = "D"
            if feature == "P6":
                # 24 leading bins and 48 isolated gaps are exactly permitted.
                masks[0, np.r_[0:24, 26:122:2]] = "G"
                quality.loc[0, "historical_raw_missing_points"] = 72
                masks[1, 40:65] = "G"
                masks[2, 0:146:2] = "G"
            frame[columns(feature)] = np.where(masks == "G", -9999.0, values)
            quality["point_status"] = ["".join(row) for row in masks]
            return frame, quality

    result = rb.repair_dataset(source, outdir=output, source_adapter=FreshAdapter(source, artifact))
    assert result["total_rows_per_feature"] == 2
    assert result["config"]["gap_limits"] == {"max_missing_bins": 72, "max_consecutive_gap_bins": 24}
    assert rb.validate_repaired_dataset(output)["valid"]
    exclusions = pd.read_csv(output / "exclusions.csv")
    assert exclusions.source_row_index.tolist() == [1, 2]
    assert exclusions.excessive_gap_features.tolist() == ["P6", "P6"]
    assert exclusions.exclusion_reason.tolist() == ["excessive_gap:P6", "excessive_gap:P6"]
    assert not exclusions.empty_P6.any()
    for feature in rb.FEATURES:
        assert read_quality(output, feature).sample_id.tolist() == ["flare-0", "flare-3"]
    p6 = read_feature(output, "P6")[columns("P6")].to_numpy()
    np.testing.assert_array_equal(p6[0, :25], np.repeat(25., 25))
    np.testing.assert_array_equal(p6[0, 25:], np.arange(26., 289.))


def test_same_class_duplicate_has_one_representative_and_preserved_labels(tmp_path, artifact):
    source, output = tmp_path / "enhanced", tmp_path / "repaired"
    fixture_enhanced(source)
    _edit_source_metadata(source, {2: {"FlrOnset": "2014-01-02 03:00:00", "event_type": 1, "noaa_active_region": 11944}})
    rb.repair_dataset(source, outdir=output, source_adapter=Adapter(source, artifact))
    mapping = pd.read_csv(output / "row_mapping.csv")
    assert mapping.source_row_index.tolist() == [0, 1, 3]
    assert mapping.event_type.tolist() == [0, 1, 1]
    assert mapping.canonical_flare_id.nunique() == 3


def test_failed_calibration_policy_never_marks_release_complete(tmp_path, artifact):
    source, output = tmp_path / "enhanced", tmp_path / "repaired"
    fixture_enhanced(source)
    with pytest.raises(ValueError, match="calibration/source policy"):
        rb.repair_dataset(source, outdir=output, source_adapter=Adapter(source, artifact, status="failed"))
    assert json.loads((output / "build_status.json").read_text())["state"] == "failed"
    assert not (output / "dataset_manifest.json").exists()


def test_source_checksum_failure_precedes_any_adapter_read(tmp_path, artifact):
    source = tmp_path / "enhanced"
    fixture_enhanced(source)
    path = source / "1998_2025_MEMSEP_dataset_with_P4_filled.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    adapter = Adapter(source, artifact)
    with pytest.raises(ValueError, match="checksum"):
        rb.repair_dataset(source, outdir=tmp_path / "repair", source_adapter=adapter)
    assert adapter.calls == 0


def test_nondeterministic_adapter_cannot_change_values_after_cohort_scan(tmp_path, artifact):
    source, output = tmp_path / "enhanced", tmp_path / "repaired"
    fixture_enhanced(source)
    class ChangingAdapter(Adapter):
        def read_feature(self, entry, metadata_columns, rows):
            frame, q = super().read_feature(entry, metadata_columns, rows)
            if self.calls > 10:
                frame.loc[0, columns(entry["feature"])[50]] += .123
            return frame, q
    with pytest.raises(ValueError, match="values/provenance changed"):
        rb.repair_dataset(source, outdir=output, source_adapter=ChangingAdapter(source, artifact))
    assert json.loads((output / "build_status.json").read_text())["state"] == "failed"


def test_validator_rejects_claimed_source_without_passed_bank_model(tmp_path, artifact):
    source, output = tmp_path / "enhanced", tmp_path / "repaired"
    fixture_enhanced(source)
    rb.repair_dataset(source, outdir=output, source_adapter=Adapter(source, artifact))
    path = output / "proton_calibration_report.json"
    report = json.loads(path.read_text())
    key = artifact["applicability"]["source_key"].copy()
    key["source_satellite"] = "GOES-99"
    report["processed_native_keys"].append(json.dumps(key))
    path.write_text(json.dumps(report))
    mpath = output / "dataset_manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["audit_files_sha256"][path.name] = digest(path)
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unvalidated proton sources"):
        rb.validate_repaired_dataset(output)


def test_verifying_state_is_not_public_completion_and_older_outputs_are_not_mixed(tmp_path, artifact):
    source, output = tmp_path / "enhanced", tmp_path / "repaired"
    fixture_enhanced(source)
    rb.repair_dataset(source, outdir=output, source_adapter=Adapter(source, artifact))
    state = output / "build_status.json"
    saved = json.loads(state.read_text())
    saved["state"] = "verifying"
    state.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="not complete"):
        rb.validate_repaired_dataset(output)
    assert cb.validate_cleaned_dataset(output, allow_verifying=True)["valid"]
    old = tmp_path / "old"
    old.mkdir()
    (old / "dataset_manifest.json").write_text(json.dumps({"workflow": "cleaned_ten_feature_release"}))
    with pytest.raises(ValueError, match="older release"):
        rb.repair_dataset(source, outdir=old, source_adapter=Adapter(source, artifact))

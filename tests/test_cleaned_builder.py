"""Regression checks for aligned, row-wise offline cleaning of twelve features."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sep_dataset import cleaned_builder as cb


FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6", "USFLUXL", "R_VALUE")


def columns(feature):
    return [f"{feature}_{i}" for i in range(288, 0, -1)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_enhanced(root, *, missing_rows=None, all_empty=False):
    """Construct source measurements independently of the cleaning code."""
    root.mkdir()
    (root / "quality").mkdir()
    n = 4
    metadata = pd.DataFrame({
        "Unnamed: 0": [10, 20, 30, 40],
        "FlrOnset": ["1998-01-02 03:00:00", "2014-01-02 03:00:00", "2025-01-02 03:00:00", "2025-02-02 03:00:00"],
        "event_type": [0, 1, 0, 1],
        "noaa_active_region": [8124, 11944, 13944, 13950],
        **{f"metadata_{i}": [i] * n for i in range(48)},
    })
    assert len(metadata.columns) == 52
    entries = []
    for feature in FEATURES:
        values = np.tile(np.arange(1.0, 289.0), (n, 1))
        if feature == "Vx":
            values *= -1
        masks = np.full((n, 288), "O", dtype="U1")
        values[0, [0, 1, 100, 101, 287]] = -9999
        masks[0, [0, 1, 100, 101, 287]] = "G"
        if feature == "P4":
            masks[0, 20:22] = ["D", "H"]
        if feature == "P5":
            values[1] = -9999
            values[1, 100] = 0.0
            masks[1] = "S"
            masks[1, 100] = "D"
        if feature == "R_VALUE":
            values[2] = -9999
            masks[2] = "G"
            values[3] = 0.0  # Zero is a real R value, not missing.
        if feature == "F":
            values[3] = float(np.float32(9999.99))
            values[3, [10, 280]] = [5.0, 15.0]
            masks[3] = "U"  # Frozen source provenance is preserved separately.
        for row in (missing_rows or {}).get(feature, []):
            values[row] = -9999
            masks[row] = "G"
        if all_empty and feature == "R_VALUE":
            values[:] = -9999
            masks[:] = "G"
        name = f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv"
        path = root / name
        frame = pd.concat([metadata, pd.DataFrame(values, columns=columns(feature))], axis=1)
        frame.to_csv(path, index=False)
        quality = pd.DataFrame({
            "row_index": range(n), "sample_id": [f"flare-{i}" for i in range(n)],
            "origin": "native_test_archive", "source_era": "test_source_era",
            "source_method": "measured_test_data", "causality_mask": "0" * 288,
            "point_status": ["".join(row) for row in masks],
            "missing_points_after": (values == -9999).sum(axis=1),
        })
        qpath = root / "quality" / f"{feature}_quality.csv"
        quality.to_csv(qpath, index=False)
        entries.append({"feature": feature, "filename": name, "rows": n, "columns": 340,
                        "sha256": digest(path), "quality_sha256": digest(qpath),
                        "source_metadata": {"units": "test units"}, "source_records": [],
                        "audit_files_sha256": {}})
    config = {"version": "fixture_enhanced_v1", "end_date_inclusive": "2025-12-31", "features": list(FEATURES)}
    manifest = {"workflow": "enhanced_twelve_feature_research_release", "status": "complete_with_reported_unavailability",
                "config": config, "metadata_columns": list(metadata.columns),
                "total_rows_per_feature": n, "feature_files": entries}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest))
    (root / "build_config.json").write_text(json.dumps(config))
    return metadata


def read_feature(root, feature):
    return pd.read_csv(root / f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv", float_precision="round_trip")


def read_quality(root, feature):
    return pd.read_csv(root / "quality" / f"{feature}_quality.csv", dtype={"sample_id": str, "point_status": str, "source_point_status": str})


def refresh_entry(root, feature, *, quality=False):
    path = root / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    entry = next(e for e in manifest["feature_files"] if e["feature"] == feature)
    if quality:
        entry["quality_sha256"] = digest(root / "quality" / f"{feature}_quality.csv")
    else:
        entry["sha256"] = digest(root / entry["filename"])
    path.write_text(json.dumps(manifest))


def test_cleaning_fills_each_feature_and_keeps_shared_ids_and_original_measurements(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    metadata = fixture_enhanced(source)
    before = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    result = cb.clean_dataset(source, outdir=output)
    assert result["total_rows_per_feature"] == 3
    assert result["excluded_rows"] == 1
    assert cb.validate_cleaned_dataset(output)["valid"]
    for feature in FEATURES:
        frame = read_feature(output, feature)
        assert frame.shape == (3, 340)
        pd.testing.assert_frame_equal(frame.iloc[:, :52], metadata.iloc[[0, 1, 3]].reset_index(drop=True), check_dtype=False)
        values = frame[columns(feature)].to_numpy()
        assert np.isfinite(values).all() and not (values == -9999).any()
        q = read_quality(output, feature)
        assert q.sample_id.tolist() == ["flare-0", "flare-1", "flare-3"]
        assert q.row_index.tolist() == [0, 1, 2]
        assert q.source_row_index.tolist() == [0, 1, 3]
        assert q.missing_points_after.eq(0).all()
        sign = -1 if feature == "Vx" else 1
        expected = np.arange(1.0, 289.0) * sign
        expected[:2] = 3 * sign
        expected[-1] = 287 * sign
        np.testing.assert_allclose(values[0], expected)
        assert q.point_status.iloc[0][:3] == "EEO"
        assert q.point_status.iloc[0][100:102] == "II"
        assert q.point_status.iloc[0][-1] == "E"
        assert q.source_point_status.iloc[0][:3] == "GGO"
        assert q.cleaning_interpolated_points.iloc[0] == 2
        assert q.cleaning_edge_filled_points.iloc[0] == 3
    assert np.array_equal(read_feature(output, "P5")[columns("P5")].iloc[1], np.zeros(288))
    assert read_quality(output, "P5").point_status.iloc[1][100] == "D"
    assert read_quality(output, "P4").point_status.iloc[0][20:22] == "DH"
    assert read_feature(output, "Vx")[columns("Vx")].to_numpy().max() < 0
    assert read_feature(output, "R_VALUE")[columns("R_VALUE")].iloc[2].eq(0).all()
    f = read_feature(output, "F")[columns("F")].iloc[2].to_numpy()
    np.testing.assert_allclose(f[:11], 5)
    np.testing.assert_allclose(f[10:281], np.linspace(5, 15, 271))
    np.testing.assert_allclose(f[280:], 15)
    exclusions = pd.read_csv(output / "exclusions.csv")
    assert exclusions.sample_id.tolist() == ["flare-2"]
    assert exclusions.empty_R_VALUE.tolist() == [True]
    assert {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()} == before


def test_missing_row_in_any_feature_is_removed_from_all_twelve_files(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source, missing_rows={"P6": [3]})
    result = cb.clean_dataset(source, outdir=output)
    assert result["total_rows_per_feature"] == 2
    assert result["excluded_rows"] == 2
    for feature in FEATURES:
        assert read_quality(output, feature).sample_id.tolist() == ["flare-0", "flare-1"]
    dropped = pd.read_csv(output / "exclusions.csv").set_index("sample_id")
    assert set(dropped.index) == {"flare-2", "flare-3"}
    assert bool(dropped.loc["flare-3", "empty_P6"])
    assert bool(dropped.loc["flare-2", "empty_R_VALUE"])


def test_source_specific_positive_fill_codes_do_not_become_real_values(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    for feature, fill, anchors in [
        ("Vx", 99999.9, [-400.0, -300.0]),
        ("flow_speed", 99999.9, [300.0, 400.0]),
        ("proton_density", 999.99, [0.0, 10.0]),
        ("T", 9999999.0, [10000.0, 20000.0]),
    ]:
        path = source / f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv"
        frame = read_feature(source, feature)
        frame.loc[3, columns(feature)] = float(np.float32(fill))
        frame.loc[3, [columns(feature)[10], columns(feature)[280]]] = anchors
        frame.to_csv(path, index=False)
        refresh_entry(source, feature)
    cb.clean_dataset(source, outdir=output)
    for feature, low, high in [("Vx", -400, -300), ("flow_speed", 300, 400), ("proton_density", 0, 10), ("T", 10000, 20000)]:
        values = read_feature(output, feature)[columns(feature)].iloc[2].to_numpy()
        assert values.min() == low and values.max() == high
        np.testing.assert_allclose(values[10:281], np.linspace(low, high, 271))


def test_fill_only_f_row_is_excluded_without_borrowing_another_sample(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    path = source / "1998_2025_MEMSEP_dataset_with_F_filled.csv"
    frame = read_feature(source, "F")
    frame.loc[1, columns("F")] = 9999.99
    frame.to_csv(path, index=False)
    refresh_entry(source, "F")
    result = cb.clean_dataset(source, outdir=output)
    assert result["total_rows_per_feature"] == 2
    for feature in FEATURES:
        assert read_quality(output, feature).sample_id.tolist() == ["flare-0", "flare-3"]
    dropped = pd.read_csv(output / "exclusions.csv").set_index("sample_id")
    assert bool(dropped.loc["flare-1", "empty_F"])


def test_nonfinite_values_are_missing_before_shared_cohort_selection(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    p4_path = source / "1998_2025_MEMSEP_dataset_with_P4_filled.csv"
    p4 = read_feature(source, "P4")
    p4.loc[0, [columns("P4")[3], columns("P4")[10]]] = [np.nan, np.inf]
    p4.to_csv(p4_path, index=False)
    refresh_entry(source, "P4")
    p6_path = source / "1998_2025_MEMSEP_dataset_with_P6_filled.csv"
    p6 = read_feature(source, "P6")
    p6.loc[1, columns("P6")] = np.nan
    p6.to_csv(p6_path, index=False)
    refresh_entry(source, "P6")
    result = cb.clean_dataset(source, outdir=output)
    assert result["total_rows_per_feature"] == 2
    for feature in FEATURES:
        assert read_quality(output, feature).sample_id.tolist() == ["flare-0", "flare-3"]
    values = read_feature(output, "P4")[columns("P4")].iloc[0]
    assert values.iloc[3] == 4.0 and values.iloc[10] == 11.0
    assert read_quality(output, "P4").point_status.iloc[0][3] == "I"
    assert read_quality(output, "P4").point_status.iloc[0][10] == "I"
    assert cb.validate_cleaned_dataset(output)["valid"]


def test_repeat_cleaning_reuses_source_without_changing_feature_outputs(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    cb.clean_dataset(source, outdir=output)
    names = [str(p.relative_to(output)) for p in output.glob("*_filled.csv")]
    names += [str(p.relative_to(output)) for p in (output / "quality").glob("*_quality.csv")]
    before = {name: digest(output / name) for name in names}
    # Re-running enhancement refreshes its manifest timestamp even when its
    # measurement and quality files are unchanged. That must not break resume.
    manifest_path = source / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_utc"] = "refreshed without changing source measurements"
    manifest_path.write_text(json.dumps(manifest))
    cb.clean_dataset(source, outdir=output)
    assert {name: digest(output / name) for name in names} == before
    assert cb.validate_cleaned_dataset(output)["valid"]


def test_refuse_source_directory_as_destination(tmp_path):
    source = tmp_path / "enhanced"
    fixture_enhanced(source)
    before = digest(source / "dataset_manifest.json")
    with pytest.raises(ValueError):
        cb.clean_dataset(source, outdir=source)
    assert digest(source / "dataset_manifest.json") == before


def test_all_samples_excluded_is_reported_as_failure_not_completed_release(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source, all_empty=True)
    with pytest.raises(ValueError):
        cb.clean_dataset(source, outdir=output)
    assert not (output / "dataset_manifest.json").exists()
    dropped = pd.read_csv(output / "exclusions.csv")
    assert len(dropped) == 4 and dropped.empty_R_VALUE.all()


def test_storage_preflight_preserves_input_and_does_not_publish_completion(tmp_path, monkeypatch):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    before = digest(source / "dataset_manifest.json")
    monkeypatch.setattr(cb.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="storage"):
        cb.clean_dataset(source, outdir=output)
    assert not (output / "dataset_manifest.json").exists()
    assert (output / "exclusions.csv").is_file()
    assert (output / "cleaning_report.json").is_file()
    assert digest(source / "dataset_manifest.json") == before


@pytest.mark.parametrize("alter", ["metadata", "sample_id"])
def test_input_alignment_errors_cannot_be_hidden_by_updated_checksums(tmp_path, alter):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    if alter == "metadata":
        path = source / "1998_2025_MEMSEP_dataset_with_P4_filled.csv"
        frame = read_feature(source, "P4")
        frame.loc[0, "event_type"] = 1
        frame.to_csv(path, index=False)
        refresh_entry(source, "P4")
    else:
        path = source / "quality" / "P4_quality.csv"
        quality = pd.read_csv(path)
        quality.loc[0, "sample_id"] = "unrelated-flare"
        quality.to_csv(path, index=False)
        refresh_entry(source, "P4", quality=True)
    with pytest.raises((ValueError, AssertionError)):
        cb.clean_dataset(source, outdir=output)
    assert not (output / "dataset_manifest.json").exists()


def test_cleaned_validator_rejects_corrupted_measurement_file(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    cb.clean_dataset(source, outdir=output)
    path = output / "1998_2025_MEMSEP_dataset_with_P4_filled.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="[Cc]hecksum"):
        cb.validate_cleaned_dataset(output)


def test_cleaned_validator_checks_sample_ids_beyond_file_checksums(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    cb.clean_dataset(source, outdir=output)
    path = output / "quality" / "P4_quality.csv"
    quality = pd.read_csv(path)
    quality.loc[0, "sample_id"] = "unrelated-flare"
    quality.to_csv(path, index=False)
    refresh_entry(output, "P4", quality=True)
    with pytest.raises((ValueError, AssertionError)):
        cb.validate_cleaned_dataset(output)


def test_cleaned_validator_checks_onsets_against_the_original_row_mapping(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    cb.clean_dataset(source, outdir=output)
    path = output / "row_mapping.csv"
    mapping = pd.read_csv(path)
    mapping.loc[0, "FlrOnset"] = "1998-01-03 03:00:00"
    mapping.to_csv(path, index=False)
    manifest_path = output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audit_files_sha256"]["row_mapping.csv"] = digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        cb.validate_cleaned_dataset(output)

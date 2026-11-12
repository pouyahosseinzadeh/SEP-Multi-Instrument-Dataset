"""Original missingness must survive the presence of finite filled baselines."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sep_dataset.compatible_baseline import FEATURES, window_columns
from sep_dataset.historical_masks import (
    _missing, anchor_digest, apply_historical_masks, build_mask_bundle,
    load_mask_bundle,
)
from sep_dataset.cleaned_builder import interpolate_row


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def originals(tmp_path):
    raw_dir = tmp_path / "original_raw"
    raw_dir.mkdir()
    metadata = pd.DataFrame({"FlrOnset": ["2005-01-02 03:00:00", "2006-01-02 03:00:00", "2013-12-31 23:00:00"],
                             "event_type": [0, 1, 0], "original_metadata": ["a", "b", "c"]})
    originals = {}
    for feature in FEATURES:
        values = np.tile(np.arange(1.0, 289.0), (3, 1))
        values[0, [0, 143, 287]] = -99999
        values[1] = -99999
        values[2, 0] = 0.0
        if feature == "Vx":
            valid = values != -99999
            values[valid] *= -1
        if feature in ("Xs", "Xl"):
            values[2, 5] = -19999.799999  # Averaged missing-code contamination.
        if feature == "F":
            values[2, 6] = float(np.float32(9999.99))
        source = pd.concat([metadata, pd.DataFrame(values, columns=window_columns(feature))], axis=1)
        path = raw_dir / feature / f"1998_2013_MEMSEP_dataset_with_{feature}.csv"
        path.parent.mkdir()
        source.to_csv(path, index=False)
        originals[feature] = source
    bundle_path = tmp_path / "historical_raw_masks.json.gz"
    before = {str(p.relative_to(raw_dir)): digest(p) for p in raw_dir.rglob("*.csv")}
    manifest = build_mask_bundle(raw_dir, bundle_path)
    assert before == {str(p.relative_to(raw_dir)): digest(p) for p in raw_dir.rglob("*.csv")}
    return raw_dir, originals, bundle_path, manifest


def filled_input(originals, feature):
    source = originals[feature].copy(deep=True)
    cols = window_columns(feature)
    values = source[cols].to_numpy(float)
    missing = _missing(values, feature)
    values[missing] = np.broadcast_to(np.linspace(.1, .9, 288), values.shape)[missing]
    source[cols] = values
    modern = pd.DataFrame({"FlrOnset": ["2025-01-01 02:00:00+00:00"], "event_type": [1], "original_metadata": ["modern"]})
    modern = pd.concat([modern, pd.DataFrame(np.arange(288.0)[None, :]+20, columns=cols)], axis=1)
    frame = pd.concat([source, modern], ignore_index=True)
    quality = pd.DataFrame({"row_index": range(4), "sample_id": ["old-a", "old-b", "old-c", "new-a"],
                            "point_status": ["U" * 288] * 3 + ["O" * 288],
                            "missing_points_after": [0] * 4, "structural_missing_points": [0] * 4,
                            "observed_points": [np.nan] * 3 + [288]})
    return frame, quality


def test_bundle_is_compact_reproducible_and_contains_provenance_without_measurements(originals, tmp_path):
    raw_dir, _, path, manifest = originals
    assert path.stat().st_size < 20000
    assert manifest["historical_rows"] == 3
    assert set(manifest["features"]) == set(FEATURES)
    assert manifest["bundle_sha256"] == digest(path)
    assert "twelve-significant" in manifest["precision"]
    again = tmp_path / "again.json.gz"
    build_mask_bundle(raw_dir, again)
    assert again.read_bytes() == path.read_bytes()
    for record in manifest["features"].values():
        assert Path(record["raw_filename"]).name == record["raw_filename"]
        assert "missing_bits" not in record
        assert "flux" not in record


def test_originally_missing_values_are_removed_before_interpolation_and_whole_missing_rows_are_exposed(originals):
    _, raw, path, _ = originals
    bundle = load_mask_bundle(path)
    frame, quality = filled_input(raw, "P6")
    before, before_q = frame.copy(deep=True), quality.copy(deep=True)
    corrected, q, report = apply_historical_masks(frame, quality, "P6", bundle)
    x = corrected[window_columns("P6")].to_numpy(float)
    assert (x[1] == -9999).all()
    assert q.point_status.iloc[1] == "G" * 288
    assert q.missing_points_after.tolist() == [3, 288, 0, 0]
    assert report["wholly_unavailable_rows"] == 1
    assert report["observed_anchor_hash_verified"] is True
    assert q.historical_raw_mask_applied.tolist() == [True, True, True, False]
    partial = x[0].copy()
    partial[partial == -9999] = np.nan
    filled, status = interpolate_row(partial, q.point_status.iloc[0])
    assert filled[0] == 2.0 and filled[143] == 144.0 and filled[287] == 287.0
    assert status[0] == "E" and status[143] == "I" and status[287] == "E"
    # Only raw-unavailable cells change, not existing finite observations.
    original_mask = _missing(raw["P6"][window_columns("P6")].to_numpy(float), "P6")
    np.testing.assert_array_equal(x[:3][~original_mask], before[window_columns("P6")].to_numpy(float)[:3][~original_mask])
    pd.testing.assert_frame_equal(corrected.iloc[3:][before.columns], before.iloc[3:])
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(quality, before_q)
    assert corrected.loc[2, "P6_288"] == 0 and q.point_status.iloc[2][0] == "U"
    assert q.observed_points.iloc[:3].isna().all()


def test_masks_align_by_utc_onset_and_label_when_full_input_rows_are_reordered(originals):
    _, raw, path, _ = originals
    frame, quality = filled_input(raw, "Vx")
    order = [2, 3, 0, 1]
    frame = frame.iloc[order].reset_index(drop=True)
    quality = quality.iloc[order].reset_index(drop=True)
    quality["row_index"] = range(len(quality))
    frame.loc[2, "FlrOnset"] = "2005-01-01T20:00:00-07:00"  # Same UTC identity.
    corrected, q, _ = apply_historical_masks(frame, quality, "Vx", load_mask_bundle(path))
    assert corrected.loc[0, "Vx_288"] == 0
    assert corrected.loc[0, "Vx_287"] == -2.0
    assert q.point_status.iloc[2][0] == "G"
    assert q.point_status.iloc[3] == "G" * 288
    assert q.historical_raw_missing_points.tolist() == [0, 0, 3, 288]


@pytest.mark.parametrize("feature", ["Xs", "Xl", "F"])
def test_unsigned_negative_contamination_and_positive_omni_fills_are_masked(feature, originals):
    _, raw, path, _ = originals
    frame, quality = filled_input(raw, feature)
    corrected, q, _ = apply_historical_masks(frame, quality, feature, load_mask_bundle(path))
    col = 6 if feature == "F" else 5
    assert corrected[window_columns(feature)].iloc[2, col] == -9999
    assert q.point_status.iloc[2][col] == "G"
    assert corrected[window_columns(feature)].iloc[2, 0] == 0


def test_float64_csv_spelling_at_float32_tie_does_not_invalidate_unchanged_observation():
    raw = np.array([[294.34584045410156, 0.0, -120.5]])
    roundtrip = np.array([[294.3458404541016, -0.0, -120.5]])
    mask = np.zeros(raw.shape, bool)
    assert anchor_digest(raw, mask) == anchor_digest(roundtrip, mask)
    changed = roundtrip.copy()
    changed[0, 0] += .001
    assert anchor_digest(raw, mask) != anchor_digest(changed, mask)


@pytest.mark.parametrize("alter", ["onset", "label", "subset", "observed_value"])
def test_unknown_historical_keys_incomplete_cohort_and_changed_anchors_fail(originals, alter):
    _, raw, path, _ = originals
    frame, quality = filled_input(raw, "P4")
    if alter == "onset":
        frame.loc[0, "FlrOnset"] = "2005-01-03 03:00"
    elif alter == "label":
        frame.loc[0, "event_type"] = 1
    elif alter == "subset":
        frame = frame.iloc[1:].reset_index(drop=True)
        quality = quality.iloc[1:].reset_index(drop=True)
        quality["row_index"] = range(len(quality))
    else:
        frame.loc[0, "P4_287"] += .1
    with pytest.raises(ValueError, match="historical cohort|anchors disagree"):
        apply_historical_masks(frame, quality, "P4", load_mask_bundle(path))


@pytest.mark.parametrize("alter", ["row_count", "row_order", "mask_length"])
def test_quality_alignment_and_point_mask_shape_fail_closed(originals, alter):
    _, raw, path, _ = originals
    frame, quality = filled_input(raw, "P4")
    if alter == "row_count":
        quality = quality.iloc[:-1]
    elif alter == "row_order":
        quality = quality.iloc[::-1]
    else:
        quality.loc[0, "point_status"] = "U" * 287
    with pytest.raises(ValueError, match="quality"):
        apply_historical_masks(frame, quality, "P4", load_mask_bundle(path))


def test_bundle_sha_and_manifest_row_count_are_checked(originals):
    _, _, path, _ = originals
    original = path.read_bytes()
    path.write_bytes(original + b"altered")
    with pytest.raises(ValueError, match="checksum"):
        load_mask_bundle(path)
    path.write_bytes(original)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["historical_rows"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row count"):
        load_mask_bundle(path)


def test_signed_velocity_and_zero_remain_valid_but_unsigned_negative_values_do_not():
    x = np.array([[-400., 0., -99999., -9999., np.nan, 1e30]])
    assert _missing(x, "Vx").tolist() == [[False, False, True, True, True, True]]
    assert _missing(x, "Xs").tolist() == [[True, False, True, True, True, True]]

"""Proton repair must replace old estimates only with tracked calibrated sources."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sep_dataset import repair_sources as rs
from sep_dataset.compatible_builder import window_index, file_sha256
from sep_dataset.proton_calibration import BANK_SCHEMA
from test_cleaned_builder import fixture_enhanced, read_feature
from test_proton_calibration import fitted


def setup_adapter(tmp_path, monkeypatch, *, unsupported=False, checksum=True):
    root = tmp_path/"enhanced"
    fixture_enhanced(root)
    source = read_feature(root, "P4")
    native, _, artifact = fitted()
    bank = tmp_path/"bank.json"
    bank.write_text(json.dumps({"schema": BANK_SCHEMA, "artifacts": [artifact]}))
    mask = tmp_path/"mask.gz"
    mask.write_bytes(b"availability-fixture")
    monkeypatch.setattr(rs, "load_mask_bundle", lambda path: {"bundle_sha256": file_sha256(mask)})
    monkeypatch.setattr(rs, "apply_historical_masks", lambda frame, q, f, bundle: (frame.copy(), q.copy(), {"feature": f, "status": "fixture"}))
    directory = root/"continuation_audit"
    directory.mkdir()
    original = json.loads((root/"dataset_manifest.json").read_text())
    for row in (2, 3):
        onset = source.FlrOnset.iloc[row]
        grid = window_index(onset)
        # One quality-valid measured spectral profile repeated as a test input.
        frame = pd.concat([native.iloc[[1]].copy() for _ in grid], ignore_index=True)
        frame.index = grid
        if unsupported and row == 2:
            frame["source_satellite"] = "GOES-18"
        path = directory/("native_sgps_"+pd.Timestamp(onset).strftime("%Y-%m")+".csv")
        frame.to_csv(path, index_label="Time")
        if checksum:
            original["feature_files"][0]["audit_files_sha256"][str(path.relative_to(root))] = file_sha256(path)
    adapter = rs.RepairAdapter(root, original, bank, mask, cache_dir=tmp_path/"cache")
    return root, original, adapter, artifact


def test_old_finite_proton_estimates_are_recomputed_with_validated_response(tmp_path, monkeypatch):
    root, manifest, adapter, _ = setup_adapter(tmp_path, monkeypatch)
    entry = next(e for e in manifest["feature_files"] if e["feature"] == "P4")
    before = file_sha256(root/entry["filename"])
    old = read_feature(root, "P4")
    frame, q = adapter.read_feature(entry, manifest["metadata_columns"], 4)
    cols = [f"P4_{i}" for i in range(288, 0, -1)]
    np.testing.assert_array_equal(frame.loc[0, cols], old.loc[0, cols])
    assert not np.array_equal(frame.loc[2, cols], old.loc[2, cols])
    assert (frame.loc[2, cols].astype(float) > 0).all()
    assert q.point_status.iloc[2] == "D"*288
    assert json.loads(q.calibration_ids.iloc[2])
    assert json.loads(q.source_calibration_keys.iloc[2])
    assert set(adapter.reports["proton_calibration_report"]["processed_native_keys"]).issubset(adapter.validated_keys)
    assert file_sha256(root/entry["filename"]) == before
    adapter.verify_sources()


def test_unsupported_native_source_quarantines_whole_window_without_other_satellite_model(tmp_path, monkeypatch):
    _, manifest, adapter, _ = setup_adapter(tmp_path, monkeypatch, unsupported=True)
    entry = next(e for e in manifest["feature_files"] if e["feature"] == "P4")
    frame, q = adapter.read_feature(entry, manifest["metadata_columns"], 4)
    assert frame.loc[2, [f"P4_{i}" for i in range(288, 0, -1)]].eq(-9999).all()
    assert q.point_status.iloc[2] == "G"*288
    assert q.unsupported_calibration_row.iloc[2]
    assert adapter.reports["proton_calibration_report"]["unsupported_rows_excluded"] == 1
    assert not q.unsupported_calibration_row.iloc[3]


def test_untracked_or_modified_native_audit_cannot_be_used(tmp_path, monkeypatch):
    _, _, adapter, _ = setup_adapter(tmp_path, monkeypatch, checksum=False)
    with pytest.raises(ValueError, match="no checksum"):
        adapter._month("2025-01")


def test_native_audit_modified_after_adapter_construction_is_rejected(tmp_path, monkeypatch):
    _, _, adapter, _ = setup_adapter(tmp_path, monkeypatch)
    path = adapter.native_paths["native_sgps_2025-01.csv"]
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="checksum"):
        adapter._month("2025-01")


def test_calibrated_cache_is_reused_only_with_exact_source_and_bank(tmp_path, monkeypatch):
    _, _, adapter, _ = setup_adapter(tmp_path, monkeypatch)
    expected = adapter._month("2025-01").copy()
    adapter.memory.clear()
    monkeypatch.setattr(rs, "apply_calibration_bank", lambda *a, **k: pytest.fail("Valid cached mapping should not be recomputed"))
    actual = adapter._month("2025-01")
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_names=False)

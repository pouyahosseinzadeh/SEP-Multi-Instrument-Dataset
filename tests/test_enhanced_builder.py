import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sep_dataset import enhanced_builder as eb
from sep_dataset.compatible_baseline import read_original_metadata
from sep_dataset.compatible_builder import _atomic_json, file_sha256, window_index


class FakeSGPS:
    def __init__(self):
        self.calls = 0
        self.source_records = [{"source": "test_sgps"}]

    def metadata(self, feature):
        return {"bin_lookahead_minutes": 5, "units": "protons/(cm^2 s sr MeV)"}

    def range(self, feature, start, stop):
        self.calls += 1
        index = pd.date_range(start, stop, freq="5min", inclusive="left")
        values = np.full(len(index), 2.0)
        if len(values) > 20:
            values[10:15] = np.nan
        return pd.Series(values, index=index, name=feature)

    def quality_range(self, feature, start, stop):
        index = pd.date_range(start, stop, freq="5min", inclusive="left")
        return pd.DataFrame({"source_product": "nativeSGPS", "source_satellite": "GOES-18"}, index=index)


class FakeMagnetic:
    source_records = [{"source": "test_magnetic"}]

    def __init__(self):
        self.calls = 0

    def metadata(self, feature):
        return {"units": "Mx" if feature == "USFLUXL" else "native log R index"}

    def window(self, feature, onset, noaa_ar):
        self.calls += 1
        grid = window_index(onset)
        value = 1e21 if feature == "USFLUXL" else 0.0
        values = np.full(288, value)
        values[0] = np.nan
        q = pd.DataFrame({"point_status": ["A"] + ["H"] * 287, "reason": "test", "source_product": "HMI", "source_satellite": "SDO"}, index=grid)
        return pd.Series(values, index=grid), q


def fixture_legacy(root):
    root.mkdir()
    metadata = read_original_metadata().iloc[:3].copy().reset_index(drop=True)
    metadata["FlrOnset"] = ["1998-01-01 03:01:00", "2014-01-01 03:01:00", "2025-01-02 03:01:00"]
    metadata["event_type"] = [0, 1, 0]
    metadata["noaa_active_region"] = [8105, 11944, 13944]
    (root / "quality").mkdir()
    entries = []
    for feature in eb.LEGACY_FEATURES:
        values = np.ones((3, 288))
        masks = ["U"*288, "O"*288, "O"*288]
        if feature in ("P4", "P5", "P6"):
            values[2] = -9999
            values[2, 0] = 7.0  # Preserve any existing nonmissing value, even after cutoff.
            masks[2] = "O" + "S"*287
        frame = pd.concat([metadata, pd.DataFrame(values, columns=eb.feature_columns(feature))], axis=1)
        filename = f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv"
        frame.to_csv(root / filename, index=False)
        audits = [{"row_index": i, "sample_id": f"sample-{i}", "origin": "baseline_frozen" if i == 0 else "extension", "point_status": m, "causality_mask": "0"*288, "source_era": "legacy", "unavailable_channel": "S" in m, **eb._counts(m)} for i, m in enumerate(masks)]
        pd.DataFrame(audits).to_csv(root / "quality" / f"{feature}_quality.csv", index=False)
        entries.append({"feature": feature, "filename": filename, "sha256": file_sha256(root / filename)})
    manifest = {"config": {"end_date_inclusive": "2025-12-31"}, "feature_files": entries, "created_utc": "original"}
    _atomic_json(root / "dataset_manifest.json", manifest)
    return metadata, entries


def test_twelve_feature_build_preserves_old_data_and_records_derived_values(tmp_path):
    root, output = tmp_path / "legacy", tmp_path / "enhanced"
    metadata, entries = fixture_legacy(root)
    before = {r["filename"]: file_sha256(root / r["filename"]) for r in entries}
    sgps, mag = FakeSGPS(), FakeMagnetic()
    result = eb.enhance_dataset(root, outdir=output, cache_dir=tmp_path / "cache", verify_legacy=False, sgps_store=sgps, magnetic_store=mag)
    assert len(result["feature_files"]) == 12
    assert eb.validate_enhanced_dataset(output)["valid"]
    for feature in eb.FEATURES:
        p = output / f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv"
        frame = pd.read_csv(p, float_precision="round_trip")
        assert frame.shape == (3, 340)
        pd.testing.assert_frame_equal(frame.iloc[:, :52], metadata, check_dtype=False, check_exact=True)
    assert all(file_sha256(root / name) == digest for name, digest in before.items())
    assert file_sha256(root / entries[0]["filename"]) == file_sha256(output / entries[0]["filename"])
    p = pd.read_csv(output / "1998_2025_MEMSEP_dataset_with_P4_filled.csv")
    assert p.P4_288.iloc[2] == 7.0
    assert (p[eb.feature_columns("P4")].iloc[:2] == 1).all().all()
    q = pd.read_csv(output / "quality" / "P4_quality.csv", dtype={"point_status": str, "causality_mask": str})
    assert "D" in q.point_status.iloc[2] and "H" in q.point_status.iloc[2] and "S" in q.point_status.iloc[2]
    assert q.point_status.iloc[2][-1] == "H"  # Causal hold, not future integration.
    assert q.causality_mask.iloc[2][-1] == "1"
    r = pd.read_csv(output / "1998_2025_MEMSEP_dataset_with_R_VALUE_filled.csv")
    assert (r[eb.feature_columns("R_VALUE")].iloc[:, 1:] == 0).all().all()  # Native zero is valid.
    assert (output / "continuation_audit" / "P4_2025-01.csv").exists()
    calls = (sgps.calls, mag.calls)
    # The original builder refreshes manifest timestamps on every resume; this
    # must not invalidate unchanged enhanced checkpoints.
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    manifest["created_utc"] = "refreshed"
    _atomic_json(root / "dataset_manifest.json", manifest)
    eb.enhance_dataset(root, outdir=output, verify_legacy=False, sgps_store=sgps, magnetic_store=mag)
    assert (sgps.calls, mag.calls) == calls
    ppath = output / "1998_2025_MEMSEP_dataset_with_P4_filled.csv"
    ppath.write_text(ppath.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        eb.validate_enhanced_dataset(output)
    eb.enhance_dataset(root, outdir=output, verify_legacy=False, sgps_store=sgps, magnetic_store=mag)
    assert eb.validate_enhanced_dataset(output)["valid"]


def test_refuse_to_overwrite_original_directory(tmp_path):
    root = tmp_path / "legacy"
    fixture_legacy(root)
    with pytest.raises(ValueError, match="own directory"):
        eb.enhance_dataset(root, outdir=root)


def test_refuse_entirely_unavailable_magnetic_feature():
    metadata = read_original_metadata().iloc[:1].copy()
    store = FakeMagnetic()
    store.window = lambda f, o, a: (pd.Series(np.nan, index=window_index(o)), pd.DataFrame(index=window_index(o)))
    with pytest.raises(RuntimeError, match="No valid"):
        eb._magnetic_feature("R_VALUE", metadata, store)


def test_missing_source_audit_is_rejected_then_repaired_on_resume(tmp_path):
    root,output=tmp_path/'legacy',tmp_path/'enhanced'
    fixture_legacy(root)
    sgps,mag=FakeSGPS(),FakeMagnetic()
    kwargs=dict(outdir=output,verify_legacy=False,sgps_store=sgps,magnetic_store=mag)
    eb.enhance_dataset(root,**kwargs)
    audit=output/'continuation_audit'/'P4_2025-01.csv'
    audit.unlink()
    with pytest.raises(ValueError,match='source audit'):
        eb.validate_enhanced_dataset(output)
    calls=sgps.calls
    eb.enhance_dataset(root,**kwargs)
    assert sgps.calls>calls and audit.exists()
    assert eb.validate_enhanced_dataset(output)['valid']

import json

import numpy as np
import pandas as pd
import pytest

from sep_dataset import compatible_builder as builder
from sep_dataset.compatible_baseline import FEATURES, window_columns


def test_interrupted_build_records_status_and_never_claims_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "read_original_metadata", lambda: _metadata())

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(builder, "_load_extension", interrupted)
    output = tmp_path / "output"
    with pytest.raises(KeyboardInterrupt):
        builder.build_compatible_dataset(outdir=output, cache_dir=tmp_path / "cache")
    status = json.loads((output / "build_status.json").read_text())
    assert status["state"] == "interrupted"
    assert "KeyboardInterrupt" in status["error"]
    assert not (output / "dataset_manifest.json").exists()


def _metadata():
    frame = pd.DataFrame({
        "Unnamed: 0": [0, 1],
        "FlrOnset": ["1998-01-01 03:01:00", "2013-12-31 12:00:00"],
        "Flrmaxtime": ["1998-01-01 03:07:00", "2013-12-31 12:05:00"],
        "Flrendtime": ["1998-01-01 03:11:00", "2013-12-31 12:15:00"],
        "start_time": ["1998-01-01T03:01:00.000", "2013-12-31T12:00:00.000"],
        "peak_time": ["1998-01-01T03:07:00.000", "2013-12-31T12:05:00.000"],
        "end_time": ["1998-01-01T03:11:00.000", "2013-12-31T12:15:00.000"],
        "event_type": [0, 1],
        **{f"original_meta_{i}": [float(i), float(i + 1)] for i in range(44)},
    })
    frame["original_meta_0"] = ["unchanged historical A", "unchanged historical B"]
    assert len(frame.columns) == 52
    return frame


def _baseline(feature, metadata):
    values = np.vstack([np.arange(288, dtype=float) + 0.1234567891234567,
                        np.arange(288, dtype=float) + 1.2345678912345678])
    return pd.concat([metadata.copy(), pd.DataFrame(values, columns=window_columns(feature))], axis=1)


class FakeStore:
    def __init__(self, *args, **kwargs):
        self.source_records = []

    def metadata(self, feature):
        return {"feature": feature, "bin_lookahead_minutes": 0 if feature in ("Xs", "Xl") else 5}

    def range(self, feature, start, stop):
        index = pd.date_range(start, stop, freq="5min", inclusive="left")
        series = pd.Series(2., index=index)
        if feature in ("P4", "P5", "P6"):
            series.loc[index >= pd.Timestamp("2020-03-05T00:00:00Z")] = np.nan
        return series

    def quality_range(self, feature, start, stop):
        series = self.range(feature, start, stop)
        return pd.DataFrame({
            "channel_status": np.where(series.isna(), "outside_original_channel_coverage", "measured"),
            "source_product": "fake_native_measurement",
            "source_satellite": "fake_observatory",
        }, index=series.index)

    def integral_range(self, *args):
        pytest.fail("The orchestrated fixture supplies explicit observed labels")


def test_original_window_ceil_order_and_flare_exclusion():
    index = builder.window_index("2025-01-02T00:03:00Z")
    assert len(index) == 288
    assert index[0] == pd.Timestamp("2025-01-01T00:05:00Z")
    assert index[-1] == pd.Timestamp("2025-01-02T00:00:00Z")
    assert (np.diff(index.asi8) == pd.Timedelta(minutes=5).value).all()
    assert (index < pd.Timestamp("2025-01-02T00:03:00Z")).all()


def test_fill_window_linear_edges_and_point_status_are_exact():
    raw = np.array([np.nan, 2., np.nan, 6., np.nan])
    filled, counts, status = builder.fill_window(raw, np.zeros(5, bool), np.full(5, 999.), return_status=True)
    np.testing.assert_array_equal(filled, [2., 2., 4., 6., 6.])
    assert status == "EOIOE"
    assert counts == {"observed_points": 2, "interpolated_points": 1,
                      "edge_filled_points": 2, "median_filled_points": 0,
                      "structural_missing_points": 0, "missing_points_after": 0,
                      "unavailable_channel": False}


def test_structural_boundary_blocks_interpolation_edges_and_medians():
    filled, counts, status = builder.fill_window(
        [2., np.nan, np.nan, np.nan, np.nan], [False, False, True, False, False],
        [9., 9., 9., 8., np.nan], return_status=True,
    )
    np.testing.assert_allclose(filled, [2., 2., np.nan, 8., np.nan], equal_nan=True)
    assert status == "OESMG"
    assert counts["edge_filled_points"] == 1
    assert counts["median_filled_points"] == 1
    assert counts["structural_missing_points"] == 1
    assert counts["missing_points_after"] == 2
    assert counts["unavailable_channel"]


def test_all_structural_window_stays_missing_despite_valid_raw_and_medians():
    filled, counts, status = builder.fill_window(np.ones(288), np.ones(288, bool), np.ones(288), return_status=True)
    assert np.isnan(filled).all()
    assert counts["observed_points"] == counts["median_filled_points"] == 0
    assert counts["structural_missing_points"] == counts["missing_points_after"] == 288
    assert status == "S" * 288


def test_wholly_missing_covered_window_uses_only_frozen_historical_medians():
    filled, counts, status = builder.fill_window([np.nan, -9999., np.inf], [False] * 3, [1., 3., np.nan], return_status=True)
    np.testing.assert_allclose(filled, [1., 3., np.nan], equal_nan=True)
    assert counts["median_filled_points"] == 2
    assert counts["missing_points_after"] == 1
    assert status == "MMG"


@pytest.mark.parametrize("feature, expected_causal", [("F", 1), ("Xs", 0)])
def test_feature_build_preserves_history_and_masks_only_future_native_bin(feature, expected_causal):
    metadata = _metadata()
    baseline = _baseline(feature, metadata)
    extension = metadata.iloc[[0]].copy()
    extension["FlrOnset"] = "2025-01-02 00:03:00"

    class SpikeStore(FakeStore):
        def range(self, feature, start, stop):
            series = super().range(feature, start, stop)
            series.iloc[-1] = 999999.
            return series

    combined, quality = builder._build_feature(feature, baseline, extension, SpikeStore())
    pd.testing.assert_frame_equal(combined.iloc[:2].reset_index(drop=True), baseline, check_exact=True)
    assert list(combined.columns) == list(metadata.columns) + window_columns(feature)
    assert combined.iloc[2]["original_meta_0"] == extension.iloc[0]["original_meta_0"]
    q = quality.iloc[2]
    assert q.causality_excluded_points == expected_causal
    assert q.causality_mask == "0" * 287 + str(expected_causal)
    assert q.point_status == "O" * 287 + ("E" if expected_causal else "O")
    assert combined.iloc[2][f"{feature}_1"] == (2. if expected_causal else 999999.)
    assert quality.iloc[0].point_status == quality.iloc[1].point_status == "U" * 288
    assert quality.iloc[0].causality_mask == "U" * 288


def test_cross_coverage_window_retains_structural_last_point():
    metadata = _metadata()
    extension = metadata.iloc[[0]].copy()
    extension["FlrOnset"] = "2020-03-05 00:03:00"
    combined, quality = builder._build_feature("P4", _baseline("P4", metadata), extension, FakeStore())
    assert combined.P4_1.iloc[-1] == -9999
    assert combined.P4_2.iloc[-1] == 2
    assert quality.iloc[-1].point_status == "O" * 287 + "S"
    assert quality.iloc[-1].structural_missing_points == 1
    assert quality.iloc[-1].causality_excluded_points == 0


def test_config_retains_inclusive_end_and_rejects_unsupported_dates():
    config = builder._config("2014-01-01", "2025-12-31")
    assert config["end_date_inclusive"] == "2025-12-31"
    assert config["features"] == list(FEATURES)
    with pytest.raises(ValueError):
        builder._config("2013-12-31", "2025-12-31")
    with pytest.raises(ValueError):
        builder._config("2014-1-1", "2025-12-31")


def test_full_ten_feature_build_resume_integrity_and_legacy_date_formats(tmp_path, monkeypatch):
    metadata = _metadata()
    baseline_dir, outdir = tmp_path / "baselines", tmp_path / "output"
    baseline_dir.mkdir()
    expected = {}
    for feature in FEATURES:
        expected[feature] = _baseline(feature, metadata)
        expected[feature].to_csv(baseline_dir / f"1998_2013_MEMSEP_dataset_with_{feature}_filled.csv", index=False)
    labels = pd.concat([metadata.iloc[[0]], metadata.iloc[[1]]], ignore_index=True)
    for name in ("FlrOnset", "Flrmaxtime", "Flrendtime", "start_time", "peak_time", "end_time"):
        # Labels may mix timezone-aware new strings and legacy naive UTC text.
        labels[name] = ["2014-01-02T00:03:00+00:00", "2025-12-31 23:59:00"]
    audit = pd.DataFrame({"event_type": [0, 1], "status": ["resolved_nsep", "resolved_sep"]})
    calls = {"labels": 0, "features": []}

    def label_flare_events(*args, **kwargs):
        calls["labels"] += 1
        return labels.copy(), audit.copy()

    actual_build = builder._build_feature

    def tracked_build(feature, *args):
        calls["features"].append(feature)
        return actual_build(feature, *args)

    monkeypatch.setattr(builder, "read_original_metadata", lambda: metadata.copy())
    monkeypatch.setattr(builder, "CompatibleSourceStore", FakeStore)
    monkeypatch.setattr(builder, "fetch_goes_flares", lambda *args: labels.copy())
    monkeypatch.setattr(builder, "label_flare_events", label_flare_events)
    monkeypatch.setattr(builder, "_build_feature", tracked_build)
    kwargs = dict(outdir=outdir, cache_dir=tmp_path / "cache", baseline_dir=baseline_dir)
    manifest = builder.build_compatible_dataset(**kwargs)
    assert manifest["total_rows_per_feature"] == 4
    assert manifest["config"]["version"] == builder.VERSION
    assert calls["features"] == list(FEATURES)
    assert builder.validate_dataset(outdir)["valid"]
    for feature in FEATURES:
        path = outdir / f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv"
        exported = pd.read_csv(path, float_precision="round_trip")
        assert exported.shape == (4, 340)
        pd.testing.assert_frame_equal(exported.iloc[:2], expected[feature], check_exact=True, check_dtype=False)
        assert exported.FlrOnset.iloc[2:].tolist() == ["2014-01-02 00:03:00", "2025-12-31 23:59:00"]
        assert exported.start_time.iloc[2:].tolist() == ["2014-01-02T00:03:00.000", "2025-12-31T23:59:00.000"]
        assert pd.to_datetime(exported.FlrOnset, utc=True).notna().all()
        if feature in ("P4", "P5", "P6"):
            assert (exported[window_columns(feature)].iloc[-1] == -9999).all()
        q = pd.read_csv(outdir / "quality" / f"{feature}_quality.csv", dtype={"point_status": str, "causality_mask": str})
        assert q.causality_mask.iloc[2].startswith("0" * 287)
    builder.build_compatible_dataset(**kwargs)
    assert calls["labels"] == 1
    assert calls["features"] == list(FEATURES)
    target = outdir / "1998_2025_MEMSEP_dataset_with_F_filled.csv"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="Checksum"):
        builder.validate_dataset(outdir)
    builder.build_compatible_dataset(**kwargs)
    assert calls["features"] == list(FEATURES) + ["F"]
    assert builder.validate_dataset(outdir)["valid"]

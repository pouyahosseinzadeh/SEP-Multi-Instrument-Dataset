"""Scientific checks for per-feature, held-out time-series evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sep_dataset import benchmark as bm


def test_metrics_reconcile_with_known_contingency_table_and_rank_auc():
    # TP40/FN10/FP20/TN30: sensitivity .8, false-positive rate .4.
    y = np.r_[np.ones(50), np.zeros(50)]
    p = np.r_[np.full(40, .9), np.full(10, .1), np.full(20, .9), np.full(30, .1)]
    result = bm.classification_metrics(y, p)
    assert [result[key] for key in ("TP", "FN", "FP", "TN")] == [40, 10, 20, 30]
    assert result["TSS"] == pytest.approx(.4)
    assert result["HSS"] == pytest.approx(.4)
    assert result["Precision"] == pytest.approx(2 / 3)
    assert result["Recall"] == pytest.approx(.8)
    assert result["F1"] == pytest.approx(8 / 11)
    assert result["BalancedAccuracy"] == pytest.approx(.7)
    assert result["ROC_AUC"] == pytest.approx(.7)
    assert result["AveragePrecision"] == pytest.approx(.8 * 2 / 3 + .2 * .5)


def test_undefined_auc_and_precision_are_unavailable_not_artificial_zeros():
    result = bm.classification_metrics([0, 0, 0], [.1, .2, .3])
    assert np.isnan(result["ROC_AUC"])
    assert np.isnan(result["PR_AUC"])
    assert np.isnan(result["TSS"])
    assert np.isnan(result["Precision"])
    assert "ROC_AUC" in result["undefined_metrics"]
    # No predicted positives is a valid failure on a two-class test cohort.
    result = bm.classification_metrics([0, 1], [.1, .1])
    assert result["F1"] == 0 and result["TSS"] == 0
    assert np.isnan(result["Precision"])
    assert result["ROC_AUC"] == .5


def test_threshold_uses_validation_discrimination_and_deterministic_ties():
    result = bm.select_threshold([0, 0, 1, 1], [.1, .2, .25, .3])
    assert result["threshold"] == .25
    assert result["validation_tss"] == 1
    assert bm.classification_metrics([0, 0, 1, 1], [.1, .2, .25, .3], .5)["TSS"] == 0
    # With no ranking skill, fixed .5 wins the declared closest-.5 tie break.
    assert bm.select_threshold([0, 1], [.2, .2])["threshold"] == .5
    assert bm.select_threshold([0, 0], [.1, .2])["threshold"] == .5


def test_temporal_buffer_and_shared_episodes_are_purged_before_evaluation():
    metadata = pd.DataFrame({
        "FlrOnset": ["2014-12-01", "2014-12-31 23:00", "2015-01-01 10:00", "2015-01-03", "2016-01-01", "2019-12-31 23:00", "2020-01-01 23:00", "2020-01-03", "2021-01-01"],
        "event_type": [0, 1, 0, 1, 0, 0, 0, 0, 1],
        "sep_start": [None, "2014-12-31", None, "2014-12-31", None, None, None, None, None],
        "sep_end": [None, "2015-01-04", None, "2015-01-04", None, None, None, None, None],
    })
    result = bm.chronological_splits(metadata)
    assert result.loc[1, "split"] == result.loc[3, "split"] == "purged"
    assert result.loc[1, "calendar_partition"] == "train"
    assert result.loc[3, "calendar_partition"] == "validation"
    assert result.loc[2, "exclusion_reason"] == result.loc[6, "exclusion_reason"] == "boundary_input_and_label_buffer"
    assert result.loc[0, "split"] == "train"
    assert result.loc[4, "split"] == "validation"
    assert result.loc[8, "split"] == "test"
    dates = pd.to_datetime(result.FlrOnset, utc=True)
    for a, b in (("train", "validation"), ("validation", "test")):
        assert dates[result.split.eq(b)].min() - dates[result.split.eq(a)].max() >= pd.Timedelta(hours=30)


def test_ongoing_episode_is_purged_even_without_an_earlier_sampled_flare():
    metadata = pd.DataFrame({"FlrOnset": ["2014-01-01", "2015-01-04", "2015-03-01", "2022-01-01"],
                             "event_type": [0, 1, 0, 1],
                             "sep_start": [None, "2014-12-30", None, None],
                             "sep_end": [None, "2015-01-08", None, None]})
    result = bm.chronological_splits(metadata)
    assert result.loc[1, "split"] == "purged"
    assert result.loc[1, "exclusion_reason"] == "SEP_episode_interval_crosses_split"


def test_same_physical_flare_cannot_appear_in_multiple_classes_or_partitions():
    metadata = pd.DataFrame({"FlrOnset": ["2014-01-01", "2021-01-01"], "event_type": [0, 1]})
    with pytest.raises(ValueError, match="canonical flare"):
        bm.chronological_splits(metadata, canonical_ids=["same-flare", "same-flare"])
    with pytest.raises(ValueError, match="at least 30"):
        bm.chronological_splits(metadata, purge_hours=24)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release(root, features=bm.FEATURES):
    root.mkdir()
    (root / "quality").mkdir()
    n = 18
    dates = list(pd.date_range("2013-01-01", periods=n, freq="2D"))
    dates += list(pd.date_range("2017-01-01", periods=n, freq="2D"))
    dates += list(pd.date_range("2023-01-01", periods=n, freq="2D"))
    labels = np.tile([0, 1], 3 * n // 2)
    metadata = pd.DataFrame({"FlrOnset": pd.DatetimeIndex(dates).astype(str), "event_type": labels,
                             "sep_peak_1": np.where(labels, 0.1, np.nan),
                             "sep_start": [None] * len(dates), "sep_end": [None] * len(dates)})
    entries = []
    for j, feature in enumerate(features):
        x = 5 + labels[:, None] * 15 + np.sin(np.arange(288)[None, :] / 20) + (np.arange(len(labels)) % 3)[:, None] * .05
        if feature == "Vx":
            x = -x
        path = root / ("1998_2025_MEMSEP_dataset_with_{}_filled.csv".format(feature))
        pd.concat([metadata, pd.DataFrame(x, columns=["{}_{}".format(feature, i) for i in range(288, 0, -1)])], axis=1).to_csv(path, index=False)
        quality = pd.DataFrame({"row_index": range(len(labels)), "sample_id": ["sample-{}".format(i) for i in range(len(labels))],
                                "canonical_flare_id": ["flare-{}".format(i) for i in range(len(labels))],
                                "source_era": ["historical"] * n + ["legacy"] * n + ["modern"] * n,
                                "point_status": ["O" * 288] * len(labels)})
        qpath = root / "quality" / (feature + "_quality.csv")
        quality.to_csv(qpath, index=False)
        entries.append({"feature": feature, "filename": path.name, "sha256": _digest(path), "quality_sha256": _digest(qpath)})
    manifest = {"workflow": "repaired_ten_feature_release_v2", "metadata_columns": list(metadata.columns),
                "total_rows_per_feature": len(metadata), "feature_files": entries}
    (root / "calibration_bank.json").write_text(json.dumps({"artifacts": []}))
    manifest["audit_files_sha256"] = {"calibration_bank.json": _digest(root / "calibration_bank.json")}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest))
    return metadata


def test_old_unrepaired_release_is_rejected_before_analysis(tmp_path):
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"workflow": "cleaned_ten_feature_release"}))
    with pytest.raises(ValueError, match="repaired ten-feature"):
        bm._load_inputs(tmp_path, bm.FEATURES)


def test_actual_twenty_cases_share_one_split_export_metrics_and_resume_without_refit(tmp_path, monkeypatch):
    pytest.importorskip("sktime")
    source, out = tmp_path / "data", tmp_path / "benchmark"
    release(source)
    before = {str(p.relative_to(source)): _digest(p) for p in source.rglob("*") if p.is_file()}
    result = bm.run_benchmarks(source, out, n_estimators=3)
    assert result["case_count"] == 20
    scores = pd.read_csv(out / "univariate_test_metrics.csv")
    assert len(scores) == 20
    assert scores.groupby("model").feature.nunique().to_dict() == {m: 10 for m in bm.MODELS}
    assert scores.n.nunique() == 1 and scores.n.iloc[0] == 18
    assert scores.SEP.eq(9).all() and scores.NSEP.eq(9).all()
    assert scores.ROC_AUC.ge(.95).all()
    assert {"TSS", "HSS", "F1", "Precision", "Recall", "ROC_AUC", "PR_AUC", "AveragePrecision"}.issubset(scores)
    for model in bm.MODELS:
        identities = []
        for feature in bm.FEATURES:
            case = pd.read_csv(out / "cases" / feature / model / "test_predictions.csv")
            identities.append(case.sample_id.tolist())
        assert all(ids == identities[0] for ids in identities)
    assert len(list(out.glob("*.png"))) == len(list(out.glob("*.pdf"))) == 3
    summary = (out / "evaluation_summary.md").read_text()
    assert "9 held-out SEP samples" in summary
    assert "27 SEP samples across all calendar partitions" in summary
    assert "No proton energy or peak-flux cutoff is applied." in summary
    assert pd.read_csv(out / "split_counts.csv").set_index("partition").loc["all", "input_SEP"] == 27
    assert pd.read_csv(out / "split_exclusion_counts.csv").empty
    def cannot_refit():
        raise AssertionError("A valid prediction checkpoint must be reused")
    monkeypatch.setattr(bm, "_estimators", lambda *args: {m: cannot_refit for m in bm.MODELS})
    again = bm.run_benchmarks(source, out, n_estimators=3)
    assert again["config_signature"] == result["config_signature"]
    assert before == {str(p.relative_to(source)): _digest(p) for p in source.rglob("*") if p.is_file()}


def test_changed_case_input_invalidates_predictions_and_requires_fit(tmp_path, monkeypatch):
    pytest.importorskip("sktime")
    source, out = tmp_path / "data", tmp_path / "benchmark"
    release(source, features=("F",))
    bm.run_benchmarks(source, out, n_estimators=2, features=("F",))
    path = source / "1998_2025_MEMSEP_dataset_with_F_filled.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "F_144"] += .5
    frame.to_csv(path, index=False)
    manifest_path = source / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_files"][0]["sha256"] = _digest(path)
    manifest_path.write_text(json.dumps(manifest))
    def must_refit():
        raise RuntimeError("refit required after source change")
    monkeypatch.setattr(bm, "_estimators", lambda *args: {m: must_refit for m in bm.MODELS})
    with pytest.raises(RuntimeError, match="refit required"):
        bm.run_benchmarks(source, out, n_estimators=2, features=("F",))


def test_input_hash_and_cross_feature_labels_are_checked_before_model_fitting(tmp_path):
    source = tmp_path / "data"
    release(source, features=("F", "P4"))
    path = source / "1998_2025_MEMSEP_dataset_with_P4_filled.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "event_type"] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="checksum"):
        bm._load_inputs(source, ("F", "P4"))
    mpath = source / "dataset_manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["feature_files"][1]["sha256"] = _digest(path)
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        bm._load_inputs(source, ("F", "P4"))


def test_calibration_observation_day_embargo_covers_input_and_label_intervals():
    metadata = pd.DataFrame({"FlrOnset": ["2013-01-01", "2017-01-01 20:00", "2017-01-02 12:00", "2017-01-03 23:00", "2023-01-01"],
                             "event_type": [0, 1, 0, 1, 0]})
    result = bm.chronological_splits(metadata, calibration_dates=["2017-01-02"])
    # The label horizon reaches into Jan2, and the next row directly overlaps.
    assert result.loc[1, "split"] == result.loc[2, "split"] == "purged"
    # Jan3 23:00 has a 24h input beginning Jan2 23:00, also inside the day.
    assert result.loc[3, "split"] == "purged"
    assert result.loc[0, "split"] == "train" and result.loc[4, "split"] == "test"
    assert result.calibration_embargo.tolist() == [False, True, True, True, False]
    assert result.loc[1:3, "calibration_overlap_dates"].eq("2017-01-02").all()


def test_calibration_embargo_removes_whole_known_positive_episode():
    metadata = pd.DataFrame({"FlrOnset": ["2013-01-01", "2017-01-03", "2017-01-12", "2017-02-01", "2023-01-01"],
                             "event_type": [0, 1, 1, 0, 1],
                             "sep_start": [None, "2017-01-02", "2017-01-02", None, None],
                             "sep_end": [None, "2017-01-15", "2017-01-15", None, None]})
    result = bm.chronological_splits(metadata, calibration_dates=["2017-01-03"])
    assert result.loc[1, "split"] == result.loc[2, "split"] == "purged"
    assert result.loc[2, "exclusion_reason"] == "SEP_episode_calibration_embargo"
    assert result.loc[3, "split"] == "validation"
    counts = bm._partition_counts(result).set_index("partition")
    assert counts.loc["validation", "input_SEP"] == counts.loc["validation", "purged_SEP"] == 2
    assert counts.loc["all", "used_SEP"] == counts.loc["test", "used_SEP"] == 1
    exclusions = bm._exclusion_counts(result).set_index(["calendar_partition", "exclusion_reason"])
    assert exclusions.loc[("validation", "calibration_observation_day_embargo"), "SEP"] == 1
    assert exclusions.loc[("validation", "SEP_episode_calibration_embargo"), "SEP"] == 1
    assert exclusions.NSEP.sum() == 0


def test_calibration_bank_dates_union_all_artifacts_and_require_traceability(tmp_path):
    path = tmp_path / "calibration_bank.json"
    path.write_text(json.dumps({"artifacts": [{"provenance": {"calibration_dates": ["2017-09-01", "2024-05-10"]}},
                                              {"provenance": {"calibration_dates": ["2024-05-10", "2025-01-01"]}}]}))
    dates, sha = bm._calibration_observation_days(tmp_path, {})
    assert dates == ["2017-09-01", "2024-05-10", "2025-01-01"]
    assert sha == _digest(path)
    path.write_text(json.dumps({"artifacts": [{"provenance": {}}]}))
    with pytest.raises(ValueError, match="lacks its actual"):
        bm._calibration_observation_days(tmp_path, {})


def test_embargo_eliminating_one_class_stops_before_any_model_fit(tmp_path, monkeypatch):
    source, out = tmp_path / "data", tmp_path / "results"
    release(source, features=("F",))
    dates = [str(value.date()) for value in pd.date_range("2023-01-03", periods=9, freq="4D")]
    bank = source / "calibration_bank.json"
    bank.write_text(json.dumps({"artifacts": [{"provenance": {"calibration_dates": dates}}]}))
    manifest_path = source / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audit_files_sha256"][bank.name] = _digest(bank)
    manifest_path.write_text(json.dumps(manifest))
    def forbidden_fit():
        raise AssertionError("A single-class embargoed partition must not reach fitting")
    monkeypatch.setattr(bm, "_estimators", lambda *args: {model: forbidden_fit for model in bm.MODELS})
    with pytest.raises(ValueError, match="test requires both SEP and NSEP"):
        bm.run_benchmarks(source, out, features=("F",))
    saved = pd.read_csv(out / "splits.csv")
    assert saved[saved.split.eq("test")].event_type.eq(0).all()
    assert (out / "calibration_embargo.csv").is_file()
    assert (out / "split_counts.csv").is_file()
    assert (out / "split_exclusion_counts.csv").is_file()
    assert not list((out / "cases").glob("*"))


def test_direct_benchmark_rejects_overlapping_cross_class_predictor_windows(tmp_path, monkeypatch):
    source, out = tmp_path / "data", tmp_path / "benchmark"
    release(source, features=("F",))
    path = next(source.glob("*_with_F_filled.csv"))
    data = pd.read_csv(path)
    data.loc[0, "FlrOnset"] = "2013-01-02T23:00:00Z"
    data.to_csv(path, index=False)
    manifest_path = source / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_files"][0]["sha256"] = _digest(path)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(bm, "_estimators", lambda *args: pytest.fail("Overlapping input must not reach fitting"))
    with pytest.raises(ValueError, match="NSEP predictor windows overlap"):
        bm.run_benchmarks(source, out, features=("F",))
    assert not out.exists()

"""A ten-feature release must select samples without magnetic availability."""
import json

import numpy as np
import pandas as pd
import pytest

from sep_dataset import cli, cleaned_builder as cb, compatible_figures, paper_figures
from test_cleaned_builder import (
    FEATURES, columns, digest, fixture_enhanced, read_feature, read_quality,
    refresh_entry,
)


TEN = FEATURES[:10]
MAGNETIC = FEATURES[10:]


def snapshot(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


def test_ten_feature_cohort_retains_magnetically_missing_sample_and_drops_unavailable_f(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned_ten"
    metadata = fixture_enhanced(source, missing_rows={"F": [1]})
    # The manifest still lists all twelve features. Absent magnetic files make
    # any accidental attempt to inspect those inputs fail immediately.
    for feature in MAGNETIC:
        (source / f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv").unlink()
        (source / "quality" / f"{feature}_quality.csv").unlink()
    before = snapshot(source)

    result = cb.clean_dataset(source, outdir=output, features=TEN)

    assert result["workflow"] == "cleaned_ten_feature_release"
    assert result["source_total_rows"] == 4
    assert result["total_rows_per_feature"] == 3
    assert result["excluded_rows"] == 1
    assert [e["feature"] for e in result["feature_files"]] == list(TEN)
    assert result["config"]["features"] == list(TEN)
    assert cb.validate_cleaned_dataset(output)["feature_files"] == 10
    assert len(list(output.glob("*_filled.csv"))) == 10
    for feature in MAGNETIC:
        assert not list(output.rglob(f"*{feature}*"))

    for feature in TEN:
        frame, quality = read_feature(output, feature), read_quality(output, feature)
        assert frame.shape == (3, 340)
        pd.testing.assert_frame_equal(frame.iloc[:, :52], metadata.iloc[[0, 2, 3]].reset_index(drop=True), check_dtype=False)
        values = frame[columns(feature)].to_numpy()
        assert np.isfinite(values).all() and not (values == -9999).any()
        assert quality.sample_id.tolist() == ["flare-0", "flare-2", "flare-3"]
        assert quality.source_row_index.tolist() == [0, 2, 3]
        assert quality.row_index.tolist() == [0, 1, 2]
        # Sample2 has no R_VALUE data but has complete original measurements.
        expected = np.arange(1.0, 289.0) * (-1 if feature == "Vx" else 1)
        np.testing.assert_array_equal(values[1], expected)
        np.testing.assert_array_equal(values[0, 2:100], expected[2:100])
        assert quality.point_status.iloc[0][:3] == "EEO"
        assert quality.point_status.iloc[0][100:102] == "II"
        assert quality.point_status.iloc[0][-1] == "E"
        assert quality.missing_points_after.eq(0).all()
    assert read_quality(output, "P4").point_status.iloc[0][20:22] == "DH"
    np.testing.assert_array_equal(read_feature(output, "F").loc[2, [columns("F")[10], columns("F")[280]]].to_numpy(), [5, 15])

    exclusions = pd.read_csv(output / "exclusions.csv")
    assert exclusions.sample_id.tolist() == ["flare-1"]
    assert exclusions.empty_F.tolist() == [True]
    assert set(c for c in exclusions if c.startswith("empty_")) == {"empty_features", *(f"empty_{f}" for f in TEN)}
    assert snapshot(source) == before


def test_ten_feature_cleaning_excludes_positive_f_fill_only_row(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned_ten"
    fixture_enhanced(source)
    path = source / "1998_2025_MEMSEP_dataset_with_F_filled.csv"
    frame = read_feature(source, "F")
    frame.loc[1, columns("F")] = float(np.float32(9999.99))
    frame.to_csv(path, index=False)
    refresh_entry(source, "F")

    cb.clean_dataset(source, outdir=output, features=TEN)

    assert read_quality(output, "F").sample_id.tolist() == ["flare-0", "flare-2", "flare-3"]
    exclusions = pd.read_csv(output / "exclusions.csv")
    assert exclusions.sample_id.tolist() == ["flare-1"]
    assert exclusions.empty_F.all()


@pytest.mark.parametrize("alter", ["checksum", "features", "mask"])
def test_ten_feature_validator_detects_corruption(tmp_path, alter):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned_ten"
    fixture_enhanced(source)
    cb.clean_dataset(source, outdir=output, features=TEN)
    if alter == "checksum":
        path = output / "1998_2025_MEMSEP_dataset_with_P4_filled.csv"
        path.write_text(path.read_text() + "\n")
    elif alter == "features":
        path = output / "dataset_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["feature_files"][-1]["feature"] = "R_VALUE"
        path.write_text(json.dumps(manifest))
    else:
        path = output / "quality" / "P4_quality.csv"
        quality = pd.read_csv(path, dtype={"point_status": str})
        quality.loc[0, "point_status"] = "O" * 287
        quality.to_csv(path, index=False)
        refresh_entry(output, "P4", quality=True)
    with pytest.raises((ValueError, AssertionError)):
        cb.validate_cleaned_dataset(output)


def test_ten_feature_selection_cannot_recover_from_already_cleaned_twelve_source(tmp_path):
    source, cleaned, output = tmp_path / "enhanced", tmp_path / "cleaned_twelve", tmp_path / "cleaned_ten"
    fixture_enhanced(source)
    cb.clean_dataset(source, outdir=cleaned)
    before = snapshot(cleaned)
    with pytest.raises(ValueError, match="(?i)(archive|enhanced|cleaned|recover)"):
        cb.clean_dataset(cleaned, outdir=output, features=TEN)
    assert snapshot(cleaned) == before
    assert not (output / "dataset_manifest.json").exists()


def test_ten_feature_cleaning_refuses_to_mix_with_existing_twelve_release(tmp_path):
    source, output = tmp_path / "enhanced", tmp_path / "cleaned"
    fixture_enhanced(source)
    cb.clean_dataset(source, outdir=output)
    before = snapshot(output)
    with pytest.raises(ValueError, match="(?i)(outdir|different|feature|folder)"):
        cb.clean_dataset(source, outdir=output, features=TEN)
    assert snapshot(output) == before


def test_cli_original_ten_only_reuses_archive_and_selects_separate_default_folder(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_clean", lambda *args, **kw: calls.append((args, kw)))
    cli.main(["clean", "--original-ten-only"])
    args, kwargs = calls[0]
    assert args[:2] == (cli.ENHANCED_DIR, cli.CLEANED_TEN_DIR)
    assert tuple(kwargs["features"]) == TEN


def test_cli_ten_release_validation_uses_cleaned_validator(tmp_path, monkeypatch):
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"workflow": "cleaned_ten_feature_release"}))
    calls = []
    monkeypatch.setattr(cb, "validate_cleaned_dataset", lambda root: calls.append(root) or {"valid": True})
    cli.main(["validate", "--dataset-dir", str(tmp_path)])
    assert calls == [tmp_path]


def test_both_plotters_recognize_ten_release_cleaning_audit(tmp_path):
    from test_ten_feature_plot_cli import existing_release

    source, output = tmp_path / "archive", tmp_path / "cleaned_ten"
    existing_release(source)
    path = source / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["workflow"] = "enhanced_twelve_feature_research_release"
    path.write_text(json.dumps(manifest))
    cb.clean_dataset(source, outdir=output, features=TEN)

    profiles = compatible_figures.run_compatible_figures(output, tmp_path / "profiles", features=("Vx",), bands=(5,))
    diagnostics = paper_figures.run_paper_figures(output, tmp_path / "paper", features=("Vx",))
    profile_manifest = json.loads((tmp_path / "profiles" / "figures_manifest.json").read_text())
    paper_manifest = json.loads((tmp_path / "paper" / "paper_figures_manifest.json").read_text())
    assert profile_manifest["input_workflow"] == paper_manifest["input_workflow"] == "cleaned_ten_feature_release"
    assert "cleaning_policy" in profile_manifest and "cleaning_policy" in paper_manifest
    coverage = pd.read_csv(tmp_path / "paper" / "feature_year_coverage.csv")
    assert "fraction_missing_before_offline_cleaning_retained_rows" in coverage
    assert profiles["outdir"] and diagnostics["outdir"]

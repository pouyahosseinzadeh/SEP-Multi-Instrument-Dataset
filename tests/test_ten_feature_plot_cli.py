"""Plot the ten retained features from existing twelve-feature releases offline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sep_dataset import cli, compatible_figures, paper_figures


FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6")
EXCLUDED = ("USFLUXL", "R_VALUE")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(root):
    return {str(path.relative_to(root)): digest(path) for path in root.rglob("*") if path.is_file()}


def existing_release(root):
    """Magnetic entries remain in the manifest, but their files are absent."""
    root.mkdir()
    (root / "quality").mkdir()
    n = 4
    metadata = pd.DataFrame({
        "Unnamed: 0": range(n),
        "FlrOnset": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "event_type": [0, 0, 1, 1],
        **{f"sep_peak_{i}": [0, 0, 20, 30] for i in range(1, 6)},
        **{f"metadata_{i}": [i] * n for i in range(44)},
    })
    assert len(metadata.columns) == 52
    entries = []
    for feature in FEATURES:
        name = f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv"
        path = root / name
        values = np.tile(np.arange(1.0, 289.0), (n, 1))
        frame = pd.concat([metadata, pd.DataFrame(values, columns=compatible_figures.timeseries_columns(feature))], axis=1)
        frame.to_csv(path, index=False)
        quality = pd.DataFrame({
            "row_index": range(n), "sample_id": [f"sample-{i}" for i in range(n)],
            "source_row_index": range(n), "origin": ["baseline", "extension", "extension", "extension"],
            "point_status": ["O" * 288] * n, "causality_mask": ["0" * 288] * n,
        })
        qpath = root / "quality" / f"{feature}_quality.csv"
        quality.to_csv(qpath, index=False)
        entries.append({"feature": feature, "filename": name, "rows": n, "columns": 340,
                        "sha256": digest(path), "quality_sha256": digest(qpath)})
    for feature in EXCLUDED:
        entries.append({"feature": feature,
                        "filename": f"1998_2025_MEMSEP_dataset_with_{feature}_filled.csv",
                        "rows": n, "columns": 340, "sha256": "absent_magnetic_file",
                        "quality_sha256": "absent_magnetic_sidecar"})
    manifest = {"workflow": "cleaned_twelve_feature_release", "metadata_columns": list(metadata.columns),
                "total_rows_per_feature": n, "feature_files": entries}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest))
    return manifest


def refresh_digest(root, feature, *, quality=False):
    path = root / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    entry = next(item for item in manifest["feature_files"] if item["feature"] == feature)
    target = root / "quality" / f"{feature}_quality.csv" if quality else root / entry["filename"]
    entry["quality_sha256" if quality else "sha256"] = digest(target)
    path.write_text(json.dumps(manifest))


def capture_rendering(monkeypatch):
    calls = []
    monkeypatch.setattr(compatible_figures, "run_compatible_figures",
                        lambda root, outdir, **kw: calls.append(("profiles", Path(root), Path(outdir), kw)))
    monkeypatch.setattr(paper_figures, "run_paper_figures",
                        lambda root, outdir, **kw: calls.append(("paper", Path(root), Path(outdir), kw)))
    return calls


def test_plot_defaults_include_all_five_integral_bands():
    args = cli.build_parser().parse_args(["plot"])
    assert args.bands == [5, 10, 30, 60, 100]


@pytest.mark.parametrize("command", ["plot", "paper-plot"])
@pytest.mark.parametrize("feature", EXCLUDED)
def test_magnetic_features_cannot_be_requested_for_figures(command, feature, capsys):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args([command, "--features", feature])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert feature in message
    if command == "plot":
        assert "invalid choice" in message


@pytest.mark.parametrize("feature", EXCLUDED)
@pytest.mark.parametrize("runner", [compatible_figures.run_compatible_figures, paper_figures.run_paper_figures])
def test_python_plotting_entrypoints_reject_excluded_features(tmp_path, runner, feature):
    with pytest.raises(ValueError, match="Unknown|supported|retained|ten"):
        runner(tmp_path, tmp_path / "figures", features=(feature,))
    assert not (tmp_path / "figures").exists()


@pytest.mark.parametrize("command", ["plot", "paper-plot"])
def test_plot_existing_ten_without_magnetic_files_or_dataset_rebuild(tmp_path, monkeypatch, command):
    source, output = tmp_path / "existing", tmp_path / "figures"
    existing_release(source)
    before = source_snapshot(source)
    calls = capture_rendering(monkeypatch)

    def full_release_validation_is_inappropriate(*args):
        raise AssertionError("Plotting ten features must not invoke the twelve-feature release validator")

    monkeypatch.setattr(cli, "_validate_release", full_release_validation_is_inappropriate)
    monkeypatch.setattr(cli, "_clean", lambda *args, **kw: pytest.fail("Plotting must not clean or rebuild the datasets"))
    cli.main([command, "--dataset-dir", str(source), "--outdir", str(output)])
    assert [call[0] for call in calls] == (["profiles", "paper"] if command == "plot" else ["paper"])
    for _, root, _, kwargs in calls:
        assert root == source
        assert tuple(kwargs["features"]) == FEATURES
    if command == "plot":
        assert calls[0][3]["bands"] == [5, 10, 30, 60, 100]
    assert source_snapshot(source) == before


def test_plot_rejects_missing_selected_original_feature(tmp_path, monkeypatch):
    source = tmp_path / "existing"
    existing_release(source)
    (source / "1998_2025_MEMSEP_dataset_with_P6_filled.csv").unlink()
    calls = capture_rendering(monkeypatch)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        cli.main(["plot", "--dataset-dir", str(source), "--outdir", str(tmp_path / "figures")])
    assert not calls


def test_plot_rejects_metadata_misalignment_between_selected_features(tmp_path, monkeypatch):
    source = tmp_path / "existing"
    existing_release(source)
    path = source / "1998_2025_MEMSEP_dataset_with_P4_filled.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "event_type"] = 1
    frame.to_csv(path, index=False)
    refresh_digest(source, "P4")
    calls = capture_rendering(monkeypatch)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        cli.main(["plot", "--dataset-dir", str(source), "--features", "Vx", "P4", "--outdir", str(tmp_path / "figures")])
    assert not calls


def test_plot_rejects_misaligned_quality_sample_identifiers(tmp_path, monkeypatch):
    source = tmp_path / "existing"
    existing_release(source)
    path = source / "quality" / "P4_quality.csv"
    quality = pd.read_csv(path)
    quality.loc[0, "sample_id"] = "different-sample"
    quality.to_csv(path, index=False)
    refresh_digest(source, "P4", quality=True)
    calls = capture_rendering(monkeypatch)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        cli.main(["plot", "--dataset-dir", str(source), "--features", "Vx", "P4", "--outdir", str(tmp_path / "figures")])
    assert not calls

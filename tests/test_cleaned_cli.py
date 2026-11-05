import json
from pathlib import Path

from sep_dataset import cli, cleaned_builder


FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6", "USFLUXL", "R_VALUE")


def test_clean_command_reuses_completed_enhanced_data_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_clean", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.main(["clean"])
    args, kwargs = calls[0]
    assert args[:2] == (cli.ENHANCED_DIR, cli.CLEANED_DIR)
    assert not kwargs


def test_default_analysis_commands_select_cleaned_dataset():
    parser = cli.build_parser()
    for command in ("validate", "plot", "paper-plot"):
        assert parser.parse_args([command]).dataset_dir == cli.CLEANED_DIR


def test_original_ten_cleanup_reuses_archive_and_has_separate_default_output(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_clean", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.main(["clean", "--original-ten-only"])
    assert calls == [((cli.ENHANCED_DIR, cli.CLEANED_TEN_DIR), {"features": cli.PLOT_FEATURES})]


def test_original_ten_cleanup_respects_explicit_folders(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_clean", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.main(["clean", "--original-ten-only", "--dataset-dir", "/existing/archive", "--outdir", "/new/ten"])
    assert calls == [(("/existing/archive", "/new/ten"), {"features": cli.PLOT_FEATURES})]


def test_ten_feature_validate_dispatches_to_cleaned_validator(tmp_path, monkeypatch):
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"workflow": "cleaned_ten_feature_release"}))
    calls = []
    monkeypatch.setattr(cleaned_builder, "validate_cleaned_dataset", lambda root: calls.append(Path(root)) or {"valid": True})
    cli.main(["validate", "--dataset-dir", str(tmp_path)])
    assert calls == [tmp_path]


def test_validate_dispatches_to_cleaned_validator(tmp_path, monkeypatch):
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"workflow": "cleaned_twelve_feature_release"}))
    calls = []
    monkeypatch.setattr(cleaned_builder, "validate_cleaned_dataset", lambda root: calls.append(Path(root)) or {"valid": True})
    cli.main(["validate", "--dataset-dir", str(tmp_path)])
    assert calls == [tmp_path]


def test_plot_validates_and_uses_only_ten_cleaned_features(tmp_path, monkeypatch):
    from sep_dataset import compatible_figures, paper_figures

    source, output = tmp_path / "cleaned", tmp_path / "figures"
    source.mkdir()
    (source / "dataset_manifest.json").write_text(json.dumps({
        "workflow": "cleaned_twelve_feature_release",
        "feature_files": [{"feature": feature} for feature in FEATURES],
    }))
    calls = []
    monkeypatch.setattr(cli, "_validate_plot_inputs", lambda root, manifest, features: calls.append(("validate", Path(root), tuple(features))) or {"valid": True})
    monkeypatch.setattr(compatible_figures, "run_compatible_figures", lambda root, outdir, **kw: calls.append(("profiles", Path(root), Path(outdir), kw)))
    monkeypatch.setattr(paper_figures, "run_paper_figures", lambda root, outdir, **kw: calls.append(("paper", Path(root), Path(outdir), kw)))
    cli.main(["plot", "--dataset-dir", str(source), "--outdir", str(output)])
    assert [call[0] for call in calls] == ["validate", "profiles", "paper"]
    assert calls[1][1:3] == (source, output)
    assert calls[0][2] == cli.PLOT_FEATURES
    assert tuple(calls[1][3]["features"]) == cli.PLOT_FEATURES
    assert calls[1][3]["bands"] == [5, 10, 30, 60, 100]
    assert calls[2][1:3] == (source, output / "paper")
    assert tuple(calls[2][3]["features"]) == cli.PLOT_FEATURES

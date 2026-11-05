"""Build, validate and plot the twelve-feature research dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

LEGACY_DIR = "data/processed/compatible_1998_2025"
ENHANCED_DIR = "data/processed/enhanced_1998_2025"
CLEANED_DIR = "data/processed/cleaned_1998_2025"
FIGURE_DIR = "reports/figures_paper_10features_1998_2025"
PLOT_FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preserve the original ten features, add USFLUXL/R_VALUE and a flagged SGPS proton continuation, and produce paper figures.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build/reuse the original ten files, then create twelve enhanced feature files.")
    build.add_argument("--end-date", "--end", default="2025-12-31")
    build.add_argument("--extension-start", default="2014-01-01")
    build.add_argument("--outdir", "--output-dir", default=None, help=f"Enhanced output directory (default {ENHANCED_DIR}); with --legacy-only, the original output directory.")
    build.add_argument("--legacy-dir", default=LEGACY_DIR, help="Original ten-feature build/cache of completed feature files; retained unchanged by enhancement.")
    build.add_argument("--cache-dir", default="data/cache/compatible", help="Existing original source cache, reused on resume.")
    build.add_argument("--enhanced-cache-dir", default="data/cache/enhanced")
    build.add_argument("--baseline-dir", help="Optional folder containing your ten original historical files.")
    build.add_argument("--magnetic-csv", help="Optional native JSOC keyword CSV for offline magnetic collection; see docs/ENHANCED_DATASET.md.")
    build.add_argument("--clean-outdir", default=CLEANED_DIR, help="Final interpolated, aligned twelve-feature output directory.")
    build.add_argument("--legacy-only", action="store_true", help="Run only the unchanged original ten-feature pipeline.")
    enhance = sub.add_parser("enhance", help="Reuse an already completed original build and collect only the new data.")
    enhance.add_argument("--legacy-dir", default=LEGACY_DIR)
    enhance.add_argument("--outdir", default=ENHANCED_DIR)
    enhance.add_argument("--cache-dir", default="data/cache/enhanced")
    enhance.add_argument("--magnetic-csv")
    enhance.add_argument("--clean-outdir", default=CLEANED_DIR)
    clean = sub.add_parser("clean", help="Offline: interpolate partial feature rows and globally remove samples with an entirely unavailable feature.")
    clean.add_argument("--dataset-dir", default=ENHANCED_DIR)
    clean.add_argument("--outdir", default=CLEANED_DIR)
    plot = sub.add_parser("plot", help="Original time-series/PCA figures plus new correlations, distributions and coverage diagnostics.")
    plot.add_argument("--dataset-dir", default=CLEANED_DIR)
    plot.add_argument("--outdir", default=FIGURE_DIR)
    plot.add_argument("--bands", nargs="+", type=int, choices=[5, 10, 30, 60, 100], default=[5, 10, 30, 60, 100])
    plot.add_argument("--features", nargs="+", choices=PLOT_FEATURES, help="Original ten features only; all ten are plotted by default.")
    plot.add_argument("--tsne", action="store_true")
    plot.add_argument("--tsne-max-samples", type=int, default=3000)
    plot.add_argument("--original-figures-only", action="store_true")
    paper = sub.add_parser("paper-plot", help="Run only the new publication diagnostics, including correlations.")
    paper.add_argument("--dataset-dir", default=CLEANED_DIR)
    paper.add_argument("--outdir", default=FIGURE_DIR + "/paper")
    paper.add_argument("--features", nargs="+", choices=PLOT_FEATURES, default=PLOT_FEATURES)
    valid = sub.add_parser("validate", help="Verify all files, exact sample alignment, checksums and per-point quality masks.")
    valid.add_argument("--dataset-dir", default=CLEANED_DIR)
    return parser


def _clean(dataset_dir, outdir=CLEANED_DIR):
    from .cleaned_builder import clean_dataset
    result = clean_dataset(dataset_dir, outdir=outdir)
    print(f"[DONE] Twelve cleaned feature files: {outdir}; {result['total_rows_per_feature']} aligned samples each; no missing feature values.", flush=True)
    print(f"[REPORT] {outdir}/availability_report.txt", flush=True)
    return result


def _enhance(legacy_dir, outdir, cache_dir, magnetic_csv=None, clean_outdir=CLEANED_DIR):
    from .enhanced_builder import enhance_dataset
    result = enhance_dataset(legacy_dir, outdir=outdir, cache_dir=cache_dir, magnetic_csv=magnetic_csv)
    print(f"[DONE] Twelve enhanced feature files: {outdir}; {result['total_rows_per_feature']} samples each.", flush=True)
    print(f"[REPORT] {outdir}/availability_report.txt", flush=True)
    _clean(outdir, clean_outdir)
    return result


def _completed_manifest(root, action="plotting"):
    path = root / "dataset_manifest.json"
    if not path.is_file():
        status_path = root / "build_status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        detail = state.get("error") or state.get("state", "not completed")
        raise RuntimeError(f"Dataset build is incomplete in {root}: {detail}. Complete or resume the build before {action}. If the enhanced archive is complete, run 'python -m sep_dataset.cli clean'.")
    return json.loads(path.read_text())


def _validate_release(root, manifest):
    if manifest.get("workflow") == "cleaned_twelve_feature_release":
        from .cleaned_builder import validate_cleaned_dataset
        return validate_cleaned_dataset(root)
    if manifest.get("workflow") == "enhanced_twelve_feature_research_release":
        from .enhanced_builder import validate_enhanced_dataset
        return validate_enhanced_dataset(root)
    from .compatible_builder import validate_dataset
    return validate_dataset(root)


def _validate_plot_inputs(root, manifest, features):
    """Verify selected inputs without requiring excluded feature datasets.

    Check schema, checksums, labels, metadata alignment and quality identifiers
    before rendering. Full release validation remains a separate command.
    """
    from .compatible_builder import file_sha256
    import pandas as pd
    if not features or len(set(features)) != len(features):
        raise ValueError("Choose distinct plotting features")
    entries = {entry["feature"]: entry for entry in manifest.get("feature_files", [])}
    metadata_columns = manifest.get("metadata_columns", [])
    if len(metadata_columns) != 52 or not {"FlrOnset", "event_type"}.issubset(metadata_columns):
        raise ValueError("Plotting requires the original 52 metadata columns")
    reference_metadata, reference_ids = None, None
    for feature in features:
        if feature not in entries:
            raise ValueError(f"Dataset manifest lacks plotting feature {feature}")
        entry = entries[feature]
        matches = sorted(root.glob(f"*_with_{feature}_filled.csv"))
        if len(matches) != 1 or matches[0].name != entry.get("filename"):
            raise ValueError(f"Expected the manifest dataset file for {feature} in {root}")
        if entry.get("sha256") and file_sha256(matches[0]) != entry["sha256"]:
            raise ValueError(f"Dataset checksum mismatch for {feature}")
        quality = root / "quality" / f"{feature}_quality.csv"
        if entry.get("quality_sha256") and (not quality.is_file() or file_sha256(quality) != entry["quality_sha256"]):
            raise ValueError(f"Quality checksum mismatch for {feature}")
        expected = metadata_columns + [f"{feature}_{i}" for i in range(288, 0, -1)]
        if pd.read_csv(matches[0], nrows=0).columns.tolist() != expected:
            raise ValueError(f"Plotting schema mismatch for {feature}")
        metadata = pd.read_csv(matches[0], usecols=metadata_columns, float_precision="round_trip", low_memory=False)
        if len(metadata) != manifest.get("total_rows_per_feature", entry.get("rows")):
            raise ValueError(f"Plotting row count mismatch for {feature}")
        if not metadata.event_type.isin([0, 1]).all():
            raise ValueError(f"Invalid SEP/NSEP labels for {feature}")
        if reference_metadata is None:
            reference_metadata = metadata
        else:
            pd.testing.assert_frame_equal(reference_metadata, metadata, check_dtype=False, check_exact=True)
        if quality.is_file():
            q = pd.read_csv(quality, usecols=lambda c: c in ("row_index", "sample_id", "point_status"), dtype={"sample_id": str, "point_status": str})
            if len(q) != len(metadata) or "row_index" not in q or q.row_index.tolist() != list(range(len(metadata))):
                raise ValueError(f"Misaligned plotting quality rows for {feature}")
            if "point_status" in q and not q.point_status.str.fullmatch("[OIEMSGUDH]{288}").all():
                raise ValueError(f"Invalid plotting quality mask for {feature}")
            if "sample_id" in q:
                if reference_ids is None:
                    reference_ids = q.sample_id
                elif not reference_ids.equals(q.sample_id):
                    raise ValueError(f"Plotting sample identifiers differ across features: {feature}")
        print(f"[PLOT CHECK] {feature}: {len(metadata)} samples", flush=True)
    return {"valid": True, "feature_files": len(features)}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        from .compatible_builder import build_compatible_dataset
        legacy_dir = (args.outdir or args.legacy_dir) if args.legacy_only else args.legacy_dir
        build_compatible_dataset(end_date=args.end_date, extension_start=args.extension_start,
                                 outdir=legacy_dir, cache_dir=args.cache_dir, baseline_dir=args.baseline_dir)
        if args.legacy_only:
            print(f"[DONE] Original ten feature files: {legacy_dir}", flush=True)
        else:
            _enhance(legacy_dir, args.outdir or ENHANCED_DIR, args.enhanced_cache_dir, args.magnetic_csv, args.clean_outdir)
    elif args.command == "enhance":
        _enhance(args.legacy_dir, args.outdir, args.cache_dir, args.magnetic_csv, args.clean_outdir)
    elif args.command == "clean":
        _clean(args.dataset_dir, args.outdir)
    elif args.command == "plot":
        from .compatible_figures import run_compatible_figures
        root = Path(args.dataset_dir)
        completed = _completed_manifest(root)
        features = tuple(args.features) if args.features else PLOT_FEATURES
        _validate_plot_inputs(root, completed, features)
        run_compatible_figures(root, args.outdir, bands=args.bands, features=features,
                               tsne=args.tsne, tsne_max_samples=args.tsne_max_samples)
        if not args.original_figures_only:
            from .paper_figures import run_paper_figures
            run_paper_figures(root, Path(args.outdir) / "paper", features=features)
        print(f"[DONE] Figures and numerical tables: {args.outdir}", flush=True)
    elif args.command == "paper-plot":
        from .paper_figures import run_paper_figures
        root = Path(args.dataset_dir)
        completed = _completed_manifest(root)
        features = tuple(args.features)
        _validate_plot_inputs(root, completed, features)
        run_paper_figures(root, args.outdir, features=features)
    else:
        root = Path(args.dataset_dir)
        manifest = _completed_manifest(root, "validation")
        result = _validate_release(root, manifest)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

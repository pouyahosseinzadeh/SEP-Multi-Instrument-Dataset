"""Build, validate and plot the twelve-feature research dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

LEGACY_DIR = "data/processed/compatible_1998_2025"
ENHANCED_DIR = "data/processed/enhanced_1998_2025"
FIGURE_DIR = "reports/figures_enhanced_1998_2025"


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
    build.add_argument("--legacy-only", action="store_true", help="Run only the unchanged original ten-feature pipeline.")
    enhance = sub.add_parser("enhance", help="Reuse an already completed original build and collect only the new data.")
    enhance.add_argument("--legacy-dir", default=LEGACY_DIR)
    enhance.add_argument("--outdir", default=ENHANCED_DIR)
    enhance.add_argument("--cache-dir", default="data/cache/enhanced")
    enhance.add_argument("--magnetic-csv")
    plot = sub.add_parser("plot", help="Original time-series/PCA figures plus new correlations, distributions and coverage diagnostics.")
    plot.add_argument("--dataset-dir", default=ENHANCED_DIR)
    plot.add_argument("--outdir", default=FIGURE_DIR)
    plot.add_argument("--bands", nargs="+", type=int, choices=[10, 30, 60, 100], default=[10, 30, 60, 100])
    plot.add_argument("--features", nargs="+", choices=["Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6", "USFLUXL", "R_VALUE"])
    plot.add_argument("--tsne", action="store_true")
    plot.add_argument("--tsne-max-samples", type=int, default=3000)
    plot.add_argument("--original-figures-only", action="store_true")
    paper = sub.add_parser("paper-plot", help="Run only the new publication diagnostics, including correlations.")
    paper.add_argument("--dataset-dir", default=ENHANCED_DIR)
    paper.add_argument("--outdir", default=FIGURE_DIR + "/paper")
    valid = sub.add_parser("validate", help="Verify all files, exact sample alignment, checksums and per-point quality masks.")
    valid.add_argument("--dataset-dir", default=ENHANCED_DIR)
    return parser


def _enhance(legacy_dir, outdir, cache_dir, magnetic_csv=None):
    from .enhanced_builder import enhance_dataset
    result = enhance_dataset(legacy_dir, outdir=outdir, cache_dir=cache_dir, magnetic_csv=magnetic_csv)
    print(f"[DONE] Twelve enhanced feature files: {outdir}; {result['total_rows_per_feature']} samples each.", flush=True)
    print(f"[REPORT] {outdir}/availability_report.txt", flush=True)
    return result


def _completed_manifest(root):
    path = root / "dataset_manifest.json"
    if not path.is_file():
        status_path = root / "build_status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        detail = state.get("error") or state.get("state", "not completed")
        raise RuntimeError(f"Dataset build is incomplete in {root}: {detail}. Complete or resume the build before plotting.")
    return json.loads(path.read_text())


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
            _enhance(legacy_dir, args.outdir or ENHANCED_DIR, args.enhanced_cache_dir, args.magnetic_csv)
    elif args.command == "enhance":
        _enhance(args.legacy_dir, args.outdir, args.cache_dir, args.magnetic_csv)
    elif args.command == "plot":
        from .compatible_figures import FEATURES, EXTRA_FEATURES, run_compatible_figures
        root = Path(args.dataset_dir)
        if args.features:
            features = tuple(args.features)
        else:
            completed = _completed_manifest(root)
            features = tuple(entry["feature"] for entry in completed["feature_files"])
        manifest = root / "dataset_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("workflow") == "enhanced_twelve_feature_research_release":
            from .enhanced_builder import validate_enhanced_dataset
            validate_enhanced_dataset(root)
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
        if completed.get("workflow") == "enhanced_twelve_feature_research_release":
            from .enhanced_builder import validate_enhanced_dataset
            validate_enhanced_dataset(root)
        run_paper_figures(root, args.outdir, features=tuple(entry["feature"] for entry in completed["feature_files"]))
    else:
        root = Path(args.dataset_dir)
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        if manifest.get("workflow") == "enhanced_twelve_feature_research_release":
            from .enhanced_builder import validate_enhanced_dataset
            result = validate_enhanced_dataset(root)
        else:
            from .compatible_builder import validate_dataset
            result = validate_dataset(root)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

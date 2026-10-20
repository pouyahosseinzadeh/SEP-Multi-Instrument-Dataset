"""Commands for the original ten-feature project extended through 2025."""
from __future__ import annotations
import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preserve the original ten MEMSEP time-series files and append observed flare samples through 2025.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build ten original-layout files; report unavailable original channels.")
    build.add_argument("--end-date", "--end", default="2025-12-31", help="Inclusive final day, YYYY-MM-DD.")
    build.add_argument("--extension-start", default="2014-01-01", help="New cohort start. Historical 1998–2013 always preserved.")
    build.add_argument("--outdir", "--output-dir", default="data/processed/compatible_1998_2025")
    build.add_argument("--cache-dir", default="data/cache/compatible")
    build.add_argument("--baseline-dir", help="Optional folder containing your ten original *_filled.csv files or the published Zenodo CSVs. Otherwise download frozen originals.")
    plot = sub.add_parser("plot", help="Original per-feature time series, confidence intervals, energy bands and full-window PCA.")
    plot.add_argument("--dataset-dir", default="data/processed/compatible_1998_2025")
    plot.add_argument("--outdir", default="reports/figures_1998_2025")
    plot.add_argument("--bands", nargs="+", type=int, choices=[10, 30, 60, 100], default=[10, 30, 60, 100])
    plot.add_argument("--features", nargs="+", choices=["Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6"])
    plot.add_argument("--tsne", action="store_true", help="Also run optional t-SNE; may be slow.")
    plot.add_argument("--tsne-max-samples", type=int, default=3000)
    valid = sub.add_parser("validate", help="Verify all ten files, historical preservation, alignment and quality reports.")
    valid.add_argument("--dataset-dir", default="data/processed/compatible_1998_2025")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        from .compatible_builder import build_compatible_dataset
        result = build_compatible_dataset(end_date=args.end_date, extension_start=args.extension_start,
                                          outdir=args.outdir, cache_dir=args.cache_dir, baseline_dir=args.baseline_dir)
        print(f"[DONE] Ten feature files in {args.outdir}; {result['total_rows_per_feature']} samples each.", flush=True)
        print(f"[REPORT] {args.outdir}/availability_report.txt", flush=True)
    elif args.command == "plot":
        from .compatible_figures import FEATURES, run_compatible_figures
        run_compatible_figures(args.dataset_dir, args.outdir, bands=args.bands,
                               features=args.features or FEATURES, tsne=args.tsne,
                               tsne_max_samples=args.tsne_max_samples)
        print(f"[DONE] Figures and numerical tables: {args.outdir}", flush=True)
    else:
        from .compatible_builder import validate_dataset
        print(json.dumps(validate_dataset(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and evaluate an extended multi-instrument SEP dataset.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Fetch catalogs and HAPI time series, then build the wide dataset.")
    build.add_argument("--start", default="1998-01-01", help="Inclusive start date, YYYY-MM-DD.")
    build.add_argument("--end", default="2024-12-31", help="Inclusive end date, YYYY-MM-DD.")
    build.add_argument("--output-dir", default="data/processed")
    build.add_argument("--cache-dir", default="data/cache")
    build.add_argument("--memsep-csv", default=None, help="Optional local or remote MEMSEP CSV to include as another catalog source.")
    build.add_argument("--catalog-only", action="store_true", help="Only write raw/merged event catalogs and sample anchors.")
    build.add_argument("--anchor", choices=["start_time", "flare_time", "max_time"], default="start_time")
    build.add_argument("--negative-ratio", type=float, default=3.0, help="Quiet-time negative samples per positive event.")
    build.add_argument("--negative-cadence", default="12h", help="Candidate cadence for quiet-time negatives.")
    build.add_argument("--blackout-days", type=float, default=3.0, help="Exclude quiet-time anchors within this many days of an SEP start.")
    build.add_argument("--lookback-hours", type=float, default=24.0)
    build.add_argument("--cadence", default="5min")
    build.add_argument("--fill-value", type=float, default=-9999.0)
    build.add_argument("--dataset", action="append", dest="datasets", help="Restrict HAPI sources by name. Can be repeated.")
    build.add_argument("--no-noaa", action="store_true", help="Disable NOAA/SWPC SPE catalog.")
    build.add_argument("--no-cdaw", action="store_true", help="Disable CDAW major SEP catalog.")
    build.add_argument("--no-donki", action="store_true", help="Disable NASA DONKI SEP catalog.")
    build.add_argument("--donki-api-key", default="DEMO_KEY")

    plots = sub.add_parser("plot", help="Create PCA, missingness, coverage, and distribution plots.")
    plots.add_argument("--dataset-csv", required=True)
    plots.add_argument("--outdir", default="reports/figures")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        from .dataset import build_extended_dataset

        result = build_extended_dataset(
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            memsep_csv=args.memsep_csv,
            catalog_only=args.catalog_only,
            anchor=args.anchor,
            negative_ratio=args.negative_ratio,
            negative_cadence=args.negative_cadence,
            blackout_days=args.blackout_days,
            lookback_hours=args.lookback_hours,
            cadence=args.cadence,
            fill_value=args.fill_value,
            datasets=args.datasets,
            include_noaa=not args.no_noaa,
            include_cdaw=not args.no_cdaw,
            include_donki=not args.no_donki,
            donki_api_key=args.donki_api_key,
        )
        print(f"[DONE] Raw catalog: {result.raw_catalog_path}")
        print(f"[DONE] Merged catalog: {result.merged_catalog_path}")
        print(f"[DONE] Samples: {result.samples_path}")
        if result.dataset_path:
            print(f"[DONE] Dataset: {result.dataset_path}")
    elif args.command == "plot":
        from .analysis import run_plots

        outputs = run_plots(args.dataset_csv, args.outdir)
        for name, path in outputs.items():
            print(f"[DONE] {name}: {path}")


if __name__ == "__main__":
    main()

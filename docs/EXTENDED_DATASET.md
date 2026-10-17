# Extended SEP Dataset Pipeline

This repository now includes a Python pipeline for rebuilding the SEP dataset beyond the original MEMSEP-limited 1998-2013 interval. The default target interval is 1998-01-01 through 2024-12-31.

## Data sources

The event catalog builder merges these public sources:

- NOAA/SWPC Solar Proton Events list, using the [current NCEI table](https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/solar-proton-events/SEP%20page%20code.html) first. The legacy SWPC text file and NASA SDAC table remain fallback sources. Both explicit-year NCEI rows and legacy month-name tables are supported, including HTML with omitted closing cell tags.
- CDAW major SEP event list, which includes GOES major SEP events and associated CME/flare metadata.
- NASA DONKI SEP events through the public DONKI API.
- Optional local MEMSEP CSV rows, if supplied with `--memsep-csv`.

The time-series builder uses HAPI endpoints with local caching:

- CDAWeb `OMNI_HRO2_5MIN` for near-Earth solar wind, IMF, geomagnetic indices, and GOES proton flux where available.
- ISWA `goesp_part_flux_P5M` for GOES >10 MeV proton flux, especially useful after OMNIWeb GOES flux availability ends in 2020.

The HAPI selector checks live metadata and maps configured aliases to the available parameter names, so minor upstream naming differences are handled at runtime.
Requests follow the server's parameter order. Months completely outside each instrument's advertised coverage are skipped. In particular, the ISWA GOES P10 source begins in April 2010; earlier windows use the other sources where available.

DONKI queries are split into intervals of at most 60 calendar days to respect the API limit. Empty responses during quiet intervals are accepted, and repeated event IDs are deduplicated. Each interval is cached, so restarting a build reuses completed downloads.

## Quick start

Install dependencies on the server:

```bash
python -m pip install -r requirements.txt
```

Build event catalogs only:

```bash
python -m sep_dataset.cli build --start 1998-01-01 --end 2024-12-31 --catalog-only
```

Build the full wide dataset:

```bash
python -m sep_dataset.cli build \
  --start 1998-01-01 \
  --end 2024-12-31 \
  --output-dir data/processed \
  --cache-dir data/cache \
  --negative-ratio 3 \
  --lookback-hours 24
```

Use the original MEMSEP CSV as an additional catalog source:

```bash
python -m sep_dataset.cli build \
  --start 1998-01-01 \
  --end 2024-12-31 \
  --memsep-csv 1998_2013_MEMSEP_dataset.csv
```

Create evaluation plots:

```bash
python -m sep_dataset.cli plot \
  --dataset-csv data/processed/extended_sep_dataset_1998_2024.csv \
  --outdir reports/figures
```

Equivalent wrappers are available:

```bash
python scripts/build_extended_sep_dataset.py --start 1998-01-01 --end 2024-12-31
python scripts/run_evaluation_plots.py --dataset-csv data/processed/extended_sep_dataset_1998_2024.csv
```

## Restarting after a failed build

Update the server checkout first, then run the build from the repository root. For Bash or Zsh, this logs the actual build failure and starts plotting only after the dataset was written:

```bash
git pull --ff-only
set -o pipefail
python -u -m sep_dataset.cli build \
  --start 1998-01-01 --end 2024-12-31 \
  --memsep-csv 1998_2013_MEMSEP_dataset.csv \
  2>&1 | tee build_sep.log \
&& python -m sep_dataset.cli plot \
  --dataset-csv data/processed/extended_sep_dataset_1998_2024.csv \
  --outdir reports/figures
```

`--catalog-only` writes catalogs and sample anchors, but does not create the wide dataset required by `plot`. If a catalog is unavailable, the build stops and names the failed source rather than silently excluding it. Invalid cached NOAA error pages are refreshed automatically. The `tmux` libtinfo warning is separate from Python catalog-download or missing-file errors.

## Outputs

The build command writes:

- `sep_catalog_raw_sources.csv`: unmerged source rows.
- `sep_catalog_merged.csv`: deduplicated SEP events with source provenance.
- `sep_samples.csv`: positive SEP anchors plus quiet-time negative anchors.
- `extended_sep_dataset_YYYY_YYYY.csv`: wide ML-ready dataset with `Feature_288 ... Feature_1` columns for 24 hours at 5-minute cadence.

The plot command writes:

- yearly label coverage
- catalog source coverage
- feature missingness
- PCA projection and explained variance
- SEP versus non-SEP mean profiles
- feature correlation heatmap
- summary CSV files used by the figures

## Notes for publication use

DONKI is a research/prototyping knowledge base. NOAA/SWPC remains the official United States source for operational space weather products. Keep the generated raw source catalog and merged catalog with the final dataset so reviewers can audit provenance and deduplication choices.


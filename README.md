# SEP-Multi-Instrument-Dataset
A Multi-Instrument Time-Series Dataset for Solar Energetic Particle Event Prediction submitted to Nature Scientific Data.

Output Datasets are uploaded here: https://zenodo.org/records/17595580

## Extended 1998-2024 dataset builder

This repository now includes a server-friendly Python pipeline for rebuilding a more comprehensive SEP dataset beyond the original MEMSEP-limited interval. It merges NOAA/SWPC SPE, CDAW major SEP, NASA DONKI SEP, and optional local MEMSEP rows, then downloads/caches multi-instrument HAPI time series from CDAWeb/OMNI and ISWA GOES proton flux.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Build catalogs only:

```bash
python -m sep_dataset.cli build --start 1998-01-01 --end 2024-12-31 --catalog-only
```

Build the full wide dataset:

```bash
python -m sep_dataset.cli build \
  --start 1998-01-01 \
  --end 2024-12-31 \
  --memsep-csv 1998_2013_MEMSEP_dataset.csv \
  --output-dir data/processed \
  --cache-dir data/cache \
  --negative-ratio 3 \
  --lookback-hours 24
```

Create evaluation plots, including PCA and distribution diagnostics:

```bash
python -m sep_dataset.cli plot \
  --dataset-csv data/processed/extended_sep_dataset_1998_2024.csv \
  --outdir reports/figures
```

See `docs/EXTENDED_DATASET.md` for source notes, outputs, and publication-use cautions.

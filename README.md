# SEP-Multi-Instrument-Dataset — original ten files extended through 2025

This project preserves the **17,794 original 1998–2013 SEP/NSEP flare samples** and appends observed C/M/X flare samples from **2014 through December 31, 2025**. It produces the original ten separate feature datasets, each with **52 original metadata columns plus 288 five-minute time-series columns**. Historical rows, labels, order and feature values are preserved. New NSEP rows are observed flare samples, as in the original project.

Original files are downloaded automatically from the frozen [Zenodo record 17595580](https://zenodo.org/records/17595580), with size and checksum verification. Only its extra energy-category `Label` column is dropped to restore the original 340-column layout; the original binary `event_type` is preserved. You may supply your own ten original filled CSVs instead.

**Original P4/P5/P6 channels are unavailable from March 5, 2020 onward.** Those cells remain `-9999`, including in `_filled.csv` files. This is explicitly reported, with no replacement energy channels or manufactured measurements. Other instrumental and calibration differences are documented in [COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Install and run on the server

Use the Git checkout containing this README. Install into a dedicated virtual environment; no `sudo` or system-wide package installation is needed:

```bash
bash scripts/setup_env.sh
bash scripts/run_project.sh
```

`run_project.sh` builds, validates, then generates figures. It stops if a stage fails and saves separate logs in `logs/`. Downloads and compatible completed feature checkpoints are reused when rerunning with the same configuration. Keep the cache and outputs when resuming.

To use your original ten filled files instead of the published copies:

```bash
bash scripts/run_project.sh --baseline-dir "/path/to/All Datasets"
```

The folder must contain all ten `1998_2013_MEMSEP_dataset_with_<feature>_filled.csv` files or their published Zenodo names. Validation rejects incompatible historical metadata or layout. Python 3.8 or newer is required; select an executable with `PYTHON_BIN=python3.10 bash scripts/setup_env.sh`.

The published CSVs have tiny floating-point serialization differences from some local copies. Use `--baseline-dir` to preserve your specific original files exactly; automatic download preserves the frozen published values exactly.

## Run each stage separately

From the repository root:

```bash
source .venv/bin/activate

python -u -m sep_dataset.cli build \
  --end-date 2025-12-31 \
  --outdir data/processed/compatible_1998_2025

python -m sep_dataset.cli validate \
  --dataset-dir data/processed/compatible_1998_2025

MPLBACKEND=Agg python -u -m sep_dataset.cli plot \
  --dataset-dir data/processed/compatible_1998_2025 \
  --outdir reports/figures_1998_2025
```

Add `--baseline-dir "/path/to/All Datasets"` to the build command to use local originals. For a small connectivity/source check, use a separate output directory and `--extension-start 2025-01-01 --end-date 2025-01-01`; it retains every historical row and appends only the requested test day.

## Dataset outputs

`data/processed/compatible_1998_2025/` contains:

```text
1998_2025_MEMSEP_dataset_with_Vx_filled.csv
1998_2025_MEMSEP_dataset_with_flow_speed_filled.csv
1998_2025_MEMSEP_dataset_with_proton_density_filled.csv
1998_2025_MEMSEP_dataset_with_T_filled.csv
1998_2025_MEMSEP_dataset_with_F_filled.csv
1998_2025_MEMSEP_dataset_with_Xs_filled.csv
1998_2025_MEMSEP_dataset_with_Xl_filled.csv
1998_2025_MEMSEP_dataset_with_P4_filled.csv
1998_2025_MEMSEP_dataset_with_P5_filled.csv
1998_2025_MEMSEP_dataset_with_P6_filled.csv
```

All ten files share the same ordered samples, labels and metadata. Their feature blocks run from `<feature>_288` to `<feature>_1`, covering 24 hours preceding `FlrOnset`. The final bin is five minutes before onset rounded up to the five-minute grid.

Read `availability_report.txt` and `dataset_manifest.json` alongside the data. `quality/<feature>_quality.csv` records observed, interpolated, edge-filled, median-filled and unavailable points. `label_audit.csv`, `unavailable_channels.csv` and `unavailable_metadata.csv` expose unresolved observations and unavailable fields. A completed build can legitimately report missing channels.

## Figures and numerical results

`reports/figures_1998_2025/figs_gt10/`, `figs_gt30/`, `figs_gt60/` and `figs_gt100/` contain **all ten features separately**:

- Mean profiles with pointwise 95% confidence intervals and Welch tests, plus explicitly labelled BH corrections.
- Individual SEP/NSEP 24-hour windows.
- PCA of the full **288-point windows**, including scores, variance, loadings, scatter plots, scree plots and heatmaps.
- Time-dependent mean-contrast heatmaps and statistics CSVs.

The root figure folder includes yearly sample counts, feature availability, before/after quality reports where audits exist, and `figures_manifest.json`. PNG and PDF copies are saved. Entirely unavailable windows are excluded from feature statistics and PCA and counted in the reports; plotting never fills them.

Optional t-SNE is slower and disabled by default:

```bash
MPLBACKEND=Agg python -m sep_dataset.cli plot \
  --dataset-dir data/processed/compatible_1998_2025 \
  --outdir reports/figures_1998_2025_tsne \
  --tsne --tsne-max-samples 3000
```

Use `--tsne-max-samples 0` for all complete windows. These are descriptive figures, not measurements of forecast accuracy. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the complete methodology.

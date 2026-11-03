# SEP dataset project — twelve features through 2025

The default workflow now produces **twelve separate feature datasets**: the original ten, plus **USFLUXL** and **R_VALUE**. Every file has the same flare samples, SEP/NSEP labels, **52 metadata columns and 288 five-minute feature values**. The original 17,794 historical rows and labels are retained, with observed C/M/X flare samples appended through December 31, 2025.

**Coverage is reported explicitly.** Original EPEAD P4/P5/P6 measurements end in March 2020. The enhanced release adds flagged SGPS-derived differential flux estimates at 30.6, 63.1 and 165 MeV; these do not recreate the original detector response or prove cross-instrument calibration. Magnetic predictors use native MDI/HMI records with bounded causal holds. Missing, stale, ambiguous-region or quality-rejected observations can remain `-9999`. Read `availability_report.txt` before interpreting results.

## Run from your latest downloaded project folder

No patch command or new clone is required. Open a terminal in the folder containing this README.

**1. Set up the project environment.** Python 3.8 or newer is required; packages are isolated from system Python.

```bash
bash scripts/setup_env.sh
```

**2. Build the twelve datasets through 2025.** Completed compatible outputs and cached downloads are reused.

```bash
.venv/bin/python -u -m sep_dataset.cli build --end-date 2025-12-31
```

**3. Validate the completed files.**

```bash
.venv/bin/python -m sep_dataset.cli validate
```

**4. Generate all figures.** This includes the original full-window PCA/profile figures and the new publication diagnostics.

```bash
.venv/bin/python -u -m sep_dataset.cli plot
```

The main build may need substantial archive downloads. Leave caches in place when resuming. A failed network request is surfaced as an error; unavailable native coverage is reported rather than replaced with invented values.

## Reuse an already completed ten-feature run

If `data/processed/compatible_1998_2025` already contains your completed original ten files, collect only the enhancements:

```bash
.venv/bin/python -u -m sep_dataset.cli enhance \
  --legacy-dir data/processed/compatible_1998_2025 \
  --outdir data/processed/enhanced_1998_2025
```

Then run validation and figures as above. The compatible input folder is retained unchanged. To keep the original strict ten-feature workflow only, use `build --legacy-only`; its scientific policy remains documented in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

The original 1998–2013 files are downloaded and checksum-verified automatically from [Zenodo record 17595580](https://zenodo.org/records/17595580). To use your local original filled files, add `--baseline-dir "/path/to/All Datasets"` to `build`. Only the published extra `Label` column is removed to restore the original layout.

## Where the datasets are

`data/processed/enhanced_1998_2025/` contains these files, each with 340 columns:

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
1998_2025_MEMSEP_dataset_with_USFLUXL_filled.csv
1998_2025_MEMSEP_dataset_with_R_VALUE_filled.csv
```

Columns run from `<feature>_288` through `<feature>_1`: 24 hours to five minutes before the five-minute-aligned `FlrOnset`. All original ten historical feature blocks are preserved; modern proton additions replace only the explicitly unavailable continuation cells in the separate enhanced output. The two magnetic feature blocks are newly collected.

`quality/<feature>_quality.csv` preserves per-point provenance. In particular, `D` identifies SGPS-derived estimates and `H` a bounded causal hold. Historical imputation provenance may be unknown (`U`). A `_filled.csv` filename does not imply that every value is observed or that structural gaps have been filled.

## Where the figures are

`reports/figures_enhanced_1998_2025/` contains:

- `figs_gt10/`, `figs_gt30/`, `figs_gt60/`, `figs_gt100/`: separate feature profiles with pointwise 95% intervals, raw/BH-corrected Welch tests, individual windows, and PCA of all 288 time points.
- `paper/`: Pearson/Spearman feature correlations, SEP/NSEP and era comparisons, distributions, variability, observed-only sensitivity, missingness and native/derived/held coverage diagnostics.
- CSV tables of statistics, sample counts, PCA loadings/scores, and manifests describing the methods. Figures are saved as PNG and PDF.

Feature correlations use **one flare window per sample**, with pairwise counts; five-minute points are not pooled as independent observations. Positive X-ray/proton/unsigned-flux summaries also have explicit log10 variants. **R_VALUE is already logarithmic and is not logged again.** The native-only sensitivity excludes imputed, held, derived and unknown-provenance points.

To regenerate only the additional diagnostics:

```bash
.venv/bin/python -m sep_dataset.cli paper-plot \
  --dataset-dir data/processed/enhanced_1998_2025 \
  --outdir reports/figures_enhanced_1998_2025/paper
```

Optional t-SNE is available with `plot --tsne`; it is slower and disabled by default. These are descriptive dataset diagnostics, not measurements of forecasting accuracy.

For native magnetic CSV input, source policies, limitations and report definitions, see [docs/ENHANCED_DATASET.md](docs/ENHANCED_DATASET.md). The older strict ten-feature workflow remains documented separately in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

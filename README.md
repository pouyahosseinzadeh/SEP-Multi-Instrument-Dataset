# SEP dataset project — twelve features through 2025

The default workflow produces **twelve separate feature datasets**: the original ten, plus **USFLUXL** and **R_VALUE**. Final files have the same retained flare samples and SEP/NSEP labels, **52 metadata columns and 288 five-minute feature values**, with no missing feature cells. The source archive preserves the original 17,794 historical rows and the extension through December 31, 2025.

**Final filling and exclusions are explicit.** Source `-9999`, nonfinite values, and documented positive OMNI fill codes are recognized before cleaning. Partially available rows receive linear interpolation inside that row and nearest-value extension at its edges. A row with no usable measurement in any feature is removed from **all twelve files**, preserving alignment. No whole row is invented from other samples or class medians. New filled cells are flagged `I`/`E`; measured, derived, held and historical provenance remain distinct. Metadata fields can still be unavailable; the no-missing guarantee applies to the 288 feature columns. See [docs/CLEANED_DATASET.md](docs/CLEANED_DATASET.md).

Original EPEAD P4/P5/P6 measurements end in March 2020. SGPS-derived estimates at 30.6, 63.1 and 165 MeV are documented separately from native legacy measurements. Magnetic variables use MDI/HMI measurements with explicit instrument and sampling limits. Interpolation does not establish cross-instrument calibration or turn filled cells into observations.

## Already downloaded all twelve datasets

Update the project code, keeping your `data` folder. Run this **offline cleanup**, then validate and regenerate the figures:

```bash
.venv/bin/python -u -m sep_dataset.cli clean
.venv/bin/python -m sep_dataset.cli validate
.venv/bin/python -u -m sep_dataset.cli plot
```

No source download is needed for `clean`. Final files are in `data/processed/cleaned_1998_2025/`; figures are in `reports/figures_cleaned_1998_2025/`. Read `exclusions.csv`, `cohort_counts.csv` and `availability_report.txt` to see the retained population. An entire missing magnetic feature row excludes that sample from all files, so this step can substantially reduce the cohort.

## Run from your latest downloaded project folder

No patch command or new clone is required. Open a terminal in the folder containing this README.

**1. Set up the project environment.** Python 3.8 or newer is required; packages are isolated from system Python.

```bash
bash scripts/setup_env.sh
```

**2. Build and clean the twelve datasets through 2025.** Completed source outputs and cached downloads are reused.

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

The main build may need substantial archive downloads. Leave caches in place when resuming. A failed network request is surfaced as an error. Cleaning uses only values within the same pre-flare feature row; excluded samples and filled cells are reported. The source archives and the final cleaned output require separate disk space.

## Reuse an already completed ten-feature run

If `data/processed/compatible_1998_2025` already contains your completed original ten files, collect only the enhancements:

```bash
.venv/bin/python -u -m sep_dataset.cli enhance \
  --legacy-dir data/processed/compatible_1998_2025 \
  --outdir data/processed/enhanced_1998_2025
```

The command also runs final cleaning. Then validate and plot as above. To keep the original strict ten-feature workflow only, use `build --legacy-only`; its scientific policy remains documented in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

The original 1998–2013 files are downloaded and checksum-verified automatically from [Zenodo record 17595580](https://zenodo.org/records/17595580). To use your local original filled files, add `--baseline-dir "/path/to/All Datasets"` to `build`. Only the published extra `Label` column is removed to restore the original layout.

## Where the datasets are

`data/processed/cleaned_1998_2025/` contains these final files, each with 340 columns:

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

Columns run from `<feature>_288` through `<feature>_1`: 24 hours to five minutes before the five-minute-aligned `FlrOnset`. The cleaned files preserve the metadata and valid feature values of retained samples. Recognized source-fill codes are corrected, partial rows are filled, and the common exclusion rule applies to all years. The complete original ten-file build remains in `compatible_1998_2025`; the twelve-feature archive before this final cleanup remains in `enhanced_1998_2025`.

`quality/<feature>_quality.csv` preserves per-point provenance, original sample IDs and source row indices. `D` identifies SGPS-derived estimates, `H` a source causal hold, and `I`/`E` interpolation/edge extension. Before-cleaning masks and fill counts expose the additional processing. Historical prefill provenance can remain unknown (`U`). Final values are finite, but are not all native observations.

## Where the figures are

`reports/figures_cleaned_1998_2025/` contains:

- `figs_gt10/`, `figs_gt30/`, `figs_gt60/`, `figs_gt100/`: separate feature profiles with pointwise 95% intervals, raw/BH-corrected Welch tests, individual windows, and PCA of all 288 time points.
- `paper/`: Pearson/Spearman feature correlations, SEP/NSEP and era comparisons, distributions, variability, observed-only sensitivity, missingness and native/derived/held coverage diagnostics.
- CSV tables of statistics, sample counts, PCA loadings/scores, and manifests describing the methods. Figures are saved as PNG and PDF.

Feature correlations use **one flare window per sample**, with pairwise counts; five-minute points are not pooled as independent observations. Positive X-ray/proton/unsigned-flux summaries also have explicit log10 variants. **R_VALUE is already logarithmic and is not logged again.** The native-only sensitivity excludes imputed, held, derived and unknown-provenance points.

To regenerate only the additional diagnostics:

```bash
.venv/bin/python -m sep_dataset.cli paper-plot \
  --dataset-dir data/processed/cleaned_1998_2025 \
  --outdir reports/figures_cleaned_1998_2025/paper
```

Optional t-SNE is available with `plot --tsne`; it is slower and disabled by default. These are descriptive dataset diagnostics, not measurements of forecasting accuracy.

For native magnetic CSV input, source policies, limitations and report definitions, see [docs/ENHANCED_DATASET.md](docs/ENHANCED_DATASET.md). The older strict ten-feature workflow remains documented separately in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

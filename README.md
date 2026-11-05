# SEP dataset project — data and paper figures through 2025

## Regenerate figures from your existing datasets

After updating the project code, keep your existing `data` folder. From the project folder, run:

```bash
.venv/bin/python -u -m sep_dataset.cli plot \
  --dataset-dir data/processed/cleaned_1998_2025 \
  --outdir reports/figures_paper_10features_1998_2025 \
  --bands 5 10 30 60 100
```

This reads the saved datasets and regenerates figures and statistical tables. No dataset download, enhancement or cleaning is needed. The saved measurements, labels and retained sample cohort are unchanged.

All figures use only the original ten features: **Vx, flow_speed, proton_density, T, F, Xs, Xl, P4, P5 and P6**. **USFLUXL and R_VALUE are excluded**, including from correlations, coverage summaries and sample comparisons. Plotting checks only the selected feature inputs; it does not require the excluded magnetic CSVs.

Profiles and PCA are generated for five integral proton energy groups:

| Energy group | SEP selection, in addition to `event_type == 1` |
|---|---|
| >5 MeV | `sep_peak_1 >= 5` pfu; added figure group |
| >10 MeV | `sep_peak_2 >= 10` pfu |
| >30 MeV | `sep_peak_3 >= 5` pfu |
| >60 MeV | `sep_peak_4 >= 5` pfu |
| >100 MeV | `sep_peak_5 >= 5` pfu |

The four original peak cuts are unchanged. Each comparison uses all saved `event_type == 0` samples as NSEP controls. An energy threshold such as >5 MeV describes the integral channel; the peak cutoff is a separate value in pfu.

The fresh output folder is `reports/figures_paper_10features_1998_2025/`. Older figure folders are kept separately and may still contain previous magnetic plots. Omitting magnetic features from figures does not restore samples that an earlier twelve-feature cleanup excluded.

## Collect a new dataset

This separate workflow is for a new collection. It builds twelve feature files: the original ten, plus USFLUXL and R_VALUE. Paper plots use the original ten only.

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

**4. Generate the paper figures for the original ten features.** This includes the full-window PCA/profile figures and publication diagnostics.

```bash
.venv/bin/python -u -m sep_dataset.cli plot
```

The main build may need substantial archive downloads. Leave caches in place when resuming. A failed network request is surfaced as an error. Cleaning uses only values within the same pre-flare feature row; excluded samples and filled cells are reported. The source archives and the final cleaned output require separate disk space.

Final files share retained flare samples and SEP/NSEP labels, with 52 metadata columns and 288 five-minute feature values. Source `-9999`, nonfinite values and documented OMNI fill codes are recognized before cleaning. Partial rows receive interpolation and nearest-value extension at the edges. A sample with any wholly unavailable feature is removed from all twelve files. See [docs/CLEANED_DATASET.md](docs/CLEANED_DATASET.md) for the filling policy and exclusion reports.

Original EPEAD P4/P5/P6 measurements end in March 2020. The enhanced archive provides flagged SGPS-derived estimates at 30.6, 63.1 and 165 MeV. The source archive preserves the original 17,794 historical rows and appends data through December 31, 2025; the cleaned release contains the retained cohort. Instrument identity and processing provenance remain available in the source and quality reports.

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

`reports/figures_paper_10features_1998_2025/` contains:

- `figs_gt5/`, `figs_gt10/`, `figs_gt30/`, `figs_gt60/`, `figs_gt100/`: feature profiles, individual windows and PCA of all 288 time points. Mean profiles retain shaded 95% confidence intervals.
- `paper/`: Pearson/Spearman feature correlations, SEP/NSEP and era comparisons, distributions, variability, observed-only sensitivity, missingness and native/derived/held coverage diagnostics.
- CSV tables of statistics, sample counts, PCA loadings/scores, and manifests describing the methods. Welch p-values, BH q-values and source notes remain in these audit outputs. Figures are saved as PNG and PDF, with concise titles and physical units instead of method/source footers or significance-marker overlays.

Feature correlations use **one flare window per sample**, with pairwise counts; five-minute points are not pooled as independent observations. Positive X-ray/proton summaries also have explicit log10 variants. The native-only sensitivity excludes imputed, held, derived and unknown-provenance points. The magnetic features are absent from every figure and plotted summary.

To regenerate only the additional diagnostics:

```bash
.venv/bin/python -m sep_dataset.cli paper-plot \
  --dataset-dir data/processed/cleaned_1998_2025 \
  --outdir reports/figures_paper_10features_1998_2025/paper
```

Optional t-SNE is available with `plot --tsne`; it is slower and disabled by default. These are descriptive dataset diagnostics, not measurements of forecasting accuracy.

For native magnetic CSV input, source policies, limitations and report definitions, see [docs/ENHANCED_DATASET.md](docs/ENHANCED_DATASET.md). The older strict ten-feature workflow remains documented separately in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

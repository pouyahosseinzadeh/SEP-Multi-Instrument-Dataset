# SEP dataset project — data and paper figures through 2025

## Create the final ten-feature release from your existing downloads

Update the project code and keep your existing `data` and `.venv` folders. These commands reuse the completed source archive in `data/processed/enhanced_1998_2025`. They perform no source download.

From your server project folder:

```bash
cd ~/Pouya/Projects/SEP_DATASET_Paper/SEP-Multi-Instrument-Dataset
```

**1. Clean and align only the original ten features.**

```bash
.venv/bin/python -u -m sep_dataset.cli clean --original-ten-only \
  --dataset-dir data/processed/enhanced_1998_2025 \
  --outdir data/processed/cleaned_ten_1998_2025
```

**2. Validate those ten output files.**

```bash
.venv/bin/python -m sep_dataset.cli validate \
  --dataset-dir data/processed/cleaned_ten_1998_2025
```

**3. Generate the figures from that ten-feature release.**

```bash
.venv/bin/python -u -m sep_dataset.cli plot \
  --dataset-dir data/processed/cleaned_ten_1998_2025 \
  --outdir reports/figures_final_ten_1998_2025
```

The datasets are written to `data/processed/cleaned_ten_1998_2025/`. Figures and statistical tables are written to `reports/figures_final_ten_1998_2025/`. The commands keep the source archive and older outputs unchanged. `clean --original-ten-only` uses the same ten-feature output directory by default; the explicit paths above make the selected release clear.

The ten features are **Vx, flow_speed, proton_density, T, F, Xs, Xl, P4, P5 and P6**. **USFLUXL and R_VALUE do not participate in the new cohort selection, validation or plots**, and their source CSVs are not required for this cleanup. The retained sample count is calculated from the ten selected files and reported at runtime; it is not assumed to equal an earlier release.

Earlier plotting changes excluded the magnetic features from images but retained the samples already saved by twelve-feature cleaning. This new cleanup starts from the unfiltered enhanced archive and recalculates the shared cohort using only the ten selected features. Use `enhanced_1998_2025` as the input: cleaning the older `cleaned_1998_2025` subset cannot recover samples that were already removed.

The cleanup recognizes source fill codes, interpolates gaps inside partially available rows and extends the nearest valid value at each edge. A sample is removed from all ten files only when at least one of those ten feature rows has no usable value. Read `cohort_counts.csv`, `exclusions.csv` and `availability_report.txt` in the new output folder for final counts and exclusion reasons. See [docs/CLEANED_DATASET.md](docs/CLEANED_DATASET.md) for the full policy.

Validation checks file structure, alignment, checksums, quality masks and final feature completeness. Scientific validation also requires source and label checks and documentation of processing. In particular, post-2020 P4/P5/P6 include flagged SGPS-derived estimates; they are not identical native legacy-channel measurements. Preserve the quality files and source documentation with the release.

## Where the ten datasets are

`data/processed/cleaned_ten_1998_2025/` contains these ten files, each with 52 metadata columns and 288 feature values:

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

Columns run from `<feature>_288` through `<feature>_1`: 24 hours to five minutes before the five-minute-aligned `FlrOnset`. The ten files share sample identities, ordering, metadata and `event_type` labels (`0` = NSEP, `1` = SEP). Existing valid feature values and metadata are preserved for retained samples.

`quality/<feature>_quality.csv` preserves per-point provenance, original sample IDs and source row indices. `D` identifies SGPS-derived estimates, `H` a source causal hold, and `I`/`E` interpolation/edge extension. Historical prefill provenance can remain unknown (`U`), and existing median-filled source values remain flagged (`M`). Finite final values are not necessarily native observations.

## Figure contents and energy selections

All figures use only the original ten features, including correlations, coverage summaries and sample comparisons. Profiles and PCA use five integral proton energy groups:

| Energy group | SEP selection, in addition to `event_type == 1` |
|---|---|
| >5 MeV | `sep_peak_1 >= 5` pfu; added figure group |
| >10 MeV | `sep_peak_2 >= 10` pfu |
| >30 MeV | `sep_peak_3 >= 5` pfu |
| >60 MeV | `sep_peak_4 >= 5` pfu |
| >100 MeV | `sep_peak_5 >= 5` pfu |

The four original peak cuts are unchanged. Each comparison uses all saved `event_type == 0` samples as NSEP controls. An energy threshold describes the integral channel; the peak cutoff is a separate value in pfu. These plotting selections do not relabel the samples.

`reports/figures_final_ten_1998_2025/` contains:

- `figs_gt5/`, `figs_gt10/`, `figs_gt30/`, `figs_gt60/`, `figs_gt100/`: feature profiles, individual windows and PCA of all 288 time points. Mean profiles retain shaded 95% confidence intervals.
- `paper/`: Pearson/Spearman correlations, SEP/NSEP and era comparisons, distributions, variability, observed-only sensitivity, missingness and measurement-status diagnostics.
- CSV tables of statistics, sample counts, PCA loadings/scores, and methods manifests. Welch p-values, BH q-values and source notes remain in these outputs. Figures are saved as PNG and PDF, with concise titles and physical units.

Feature correlations use one flare window per sample, with pairwise counts. Five-minute points are not pooled as independent observations. Positive X-ray/proton summaries have explicit log10 variants. The observed-only sensitivity excludes filled, held, derived and unknown-provenance points. These are descriptive diagnostics, not measurements of forecasting accuracy.

To regenerate only the additional diagnostics from the new ten-feature release:

```bash
.venv/bin/python -m sep_dataset.cli paper-plot \
  --dataset-dir data/processed/cleaned_ten_1998_2025 \
  --outdir reports/figures_final_ten_1998_2025/paper
```

Optional t-SNE is available with `plot --tsne`; it is slower and disabled by default.

## Regenerate figures for the older saved cohort only

If you deliberately want figures for the earlier `cleaned_1998_2025` subset, use:

```bash
.venv/bin/python -u -m sep_dataset.cli plot \
  --dataset-dir data/processed/cleaned_1998_2025 \
  --outdir reports/figures_paper_10features_1998_2025
```

This command reads the saved subset without changing its cohort. It omits magnetic features from the figures but does not restore samples excluded during twelve-feature cleaning. Default `plot`, `paper-plot` and `validate` still select that older directory unless `--dataset-dir` is supplied. Use the explicit new ten-feature paths above for the final ten-feature release.

## Optional original collection workflow

For a new collection, set up the environment with `bash scripts/setup_env.sh` and run `.venv/bin/python -u -m sep_dataset.cli build --end-date 2025-12-31`. Python 3.8 or newer is required. This original collection workflow builds the original ten files, creates the twelve-feature enhanced archive and performs twelve-feature cleanup for compatibility. After collection, use the three commands at the top to create the separate ten-feature release.

For an already completed original build in `data/processed/compatible_1998_2025`, `enhance` reuses those files and collects the enhancements. Neither `build` nor `enhance` is needed when the enhanced source archive is already complete.

The original 1998–2013 files are downloaded and checksum-verified automatically from [Zenodo record 17595580](https://zenodo.org/records/17595580). Optional `--baseline-dir "/path/to/All Datasets"` selects local originals. `build --legacy-only` retains the strict original ten-feature source workflow. Its policies are documented in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

Original EPEAD P4/P5/P6 measurements end in March 2020. The enhanced archive provides flagged SGPS-derived estimates at 30.6, 63.1 and 165 MeV. Source and processing limitations are documented in [docs/ENHANCED_DATASET.md](docs/ENHANCED_DATASET.md). The complete original build remains in `compatible_1998_2025`; the enhanced source archive remains in `enhanced_1998_2025`.

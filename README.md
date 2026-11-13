# SEP dataset: repaired ten-feature release and model evaluation

The project collects the original ten feature datasets through December 2025, validates their common SEP/NSEP cohort, and generates paper figures and univariate model results.

## Update a completed collection

Download the [current project ZIP](https://github.com/pouyahosseinzadeh/SEP-Multi-Instrument-Dataset/archive/refs/heads/main.zip). Copy the updated `sep_dataset/` and `scripts/` into your existing server project. Keep the downloaded archive in `data/fresh_1998_2025/native/` and the two historical availability files under `calibration/`. Run from the server project folder:

```bash
.venv/bin/python -u -m sep_dataset.cli repair \
  --dataset-dir data/fresh_1998_2025/native \
  --outdir data/processed/overlap_filtered_ten_1998_2025
.venv/bin/python -m sep_dataset.cli validate
.venv/bin/python -u -m sep_dataset.cli plot --with-models --n-jobs 4
```

This offline update uses the previous collection's measurements. The new output folder keeps earlier results separate. If the native acquisition archive was deleted, follow the fresh download steps below.

## Start over from fresh downloads

1. Download the [current project ZIP](https://github.com/pouyahosseinzadeh/SEP-Multi-Instrument-Dataset/archive/refs/heads/main.zip). Copy the updated `sep_dataset/`, `scripts/`, `calibration/`, and both requirements files into your existing server project.
2. Copy `sep-historical-availability.zip` into that project folder. It provides the original 1998–2013 missingness and observed-value fingerprints. Keep this ZIP and any original unfilled CSVs. If you have the original unfilled CSVs, you can regenerate the bundle locally with `.venv/bin/python scripts/build_historical_masks.py --raw-dir /path/to/originals`.
3. From the project folder, run:

```bash
cd ~/Pouya/Projects/SEP_DATASET_Paper/SEP-Multi-Instrument-Dataset
unzip -o sep-historical-availability.zip
df -h .
bash scripts/setup_env.sh
bash scripts/run_fresh_project.sh
```

Fresh source downloads and checkpoints are in `data/fresh_1998_2025/`. The runner uses this directory independently of earlier `data/cache/` and `data/processed/` releases. You can remove those older generated outputs to reclaim space after saving any files you need. Repeat the same command to resume interrupted downloads and collection. To use a larger mounted disk, pass absolute paths via `--work-dir` and `--outdir`.

The 1998–2013 published data are downloaded and verified against Zenodo checksums. The extension collects source observations and flare labels for 2014–2025. The collector archives the native modern proton spectra needed for calibration. It collects no magnetic-region datasets. The ten validated submission CSVs and their provenance are in `data/processed/overlap_filtered_ten_1998_2025/`.

After collection finishes, run the figures and models:

```bash
.venv/bin/python -u -m sep_dataset.cli plot \
  --dataset-dir data/processed/overlap_filtered_ten_1998_2025 \
  --outdir reports/figures_overlap_filtered_ten_1998_2025 \
  --with-models --n-jobs 4
```

Logs from collection and validation are in `logs/fresh_*.log`. The figures include a yearly P4/P5/P6 distribution and coverage table, `paper/proton_flux_by_year.csv`, and a matching PDF. The held-out metrics and compact plots are under `models/`.

## Reuse an existing archive

For an earlier completed unfiltered enhanced archive, the following offline workflow remains available. Keep its ten original CSVs, manifest, quality sidecars, and native proton audits. The current fresh release uses stricter interpolation limits.

```bash
.venv/bin/python -u -m sep_dataset.cli repair \
  --dataset-dir data/processed/enhanced_1998_2025 \
  --outdir data/processed/repaired_ten_1998_2025

.venv/bin/python -m sep_dataset.cli validate \
  --dataset-dir data/processed/repaired_ten_1998_2025

.venv/bin/python -u -m sep_dataset.cli plot \
  --dataset-dir data/processed/repaired_ten_1998_2025 \
  --outdir reports/figures_repaired_ten_1998_2025 \
  --with-models --n-jobs 4
```

For later figure-only reruns, omit `--with-models`. To run only the model evaluation:

```bash
.venv/bin/python -u -m sep_dataset.cli benchmark \
  --dataset-dir data/processed/repaired_ten_1998_2025 \
  --outdir reports/figures_repaired_ten_1998_2025/models \
  --n-jobs 4
```

Matching input/configuration hashes allow completed feature outputs and model-prediction cases to resume. Changes to calibration, raw masks, data or evaluation settings invalidate the relevant checkpoints. Use a new output directory when repair reports a changed source/calibration policy.

## What the repair changes

The ten features are **Vx, flow_speed, proton_density, T, F, Xs, Xl, P4, P5 and P6**. USFLUXL and R_VALUE do not participate in selection, figures or models. Each feature file retains **52 metadata columns plus 288 five-minute pre-flare values**, from `<feature>_288` to `<feature>_1`. `event_type` is `0` for NSEP and `1` for SEP.

Historical originally missing cells are restored before interpolation. A row with no valid anchors in any selected feature is excluded from all ten files, including inherited wholly filled templates whose raw rows were unavailable. Partial observational gaps receive interpolation within their own row and nearest-value edge extension, with `I`/`E` flags. The fresh release keeps at least 216 of 288 source-supported points per feature and excludes any row with more than 24 consecutive unavailable bins. At most six hours of bins in total and two consecutive hours can be filled. Valid zeros are retained. All 288 feature values must be finite and free of missing-value codes. Unreported metadata stays blank and is counted in `metadata_availability.csv`. Existing valid anchors and retained labels are preserved. Entirely median-filled source rows are unavailable unless the source adapter replaces them with supported observations.

After the March 2020 transition, P4/P5/P6 are recomputed from native SGPS spectra using **passed, source-specific legacy-instrument-equivalent response estimates**. The reference is the original GOES-15 W-labelled **uncorrected** channel response, including its background. These are estimated continuations of that instrument quantity; they are not native EPEAD measurements or proof of uniform physical calibration across every instrument/year.

The response models use simultaneous observations and separate held-out quiet/event tests. They do not force averages from different decades to agree. Applicability depends on satellite, product, look direction, numerical energy grid and tested spectral range. Bridges identify their modeled validation target and inherit their reference model's limitations. **Unsupported or out-of-domain windows are quarantined**, rather than filled from uncalibrated modern values. Read `proton_calibration_report.json` for actual coverage. No fixed sample count or complete coverage of every year is promised.

Physical-flare identity is checked before the cohort is selected. Conflicting SEP/NSEP labels for one identified flare quarantine the entire group; same-label duplicates keep one deterministic representative. Exact UTC onset is the primary key. Nearby onsets require additional matching region and flare-timing evidence. The final files have one canonical flare identity per row and no identity shared across SEP and NSEP. Matching evidence and exclusions are recorded.

An NSEP row is also excluded from all ten files if any of its 288 five-minute predictor bins is shared with any labeled SEP candidate. Each half-open interval starts 24 hours before the five-minute boundary at or immediately after the UTC flare onset and stops at that boundary. Intervals that only touch do not share a bin. All labeled SEP candidates protect their windows, including those excluded by feature-quality rules. This rule only removes NSEP rows. `temporal_overlap_audit.csv` records an example SEP match and the number of bins shared for each flagged NSEP row.

## Figures and model results

Profiles and PCA retain five integral energy groups: **>5, >10, >30, >60 and >100 MeV**. These are metadata-based selections, not the differential P4/P5/P6 predictor names. The >5 group uses `event_type == 1` and `sep_peak_1 >= 5` pfu; the original four cuts remain unchanged. Saved `event_type == 0` samples provide the NSEP controls.

Descriptive figures include profiles, full-window PCA, Pearson/Spearman correlations, distributions, missingness and measurement-status diagnostics. Images have concise labels; statistical/source methods remain in tables and manifests.

The model stage runs **20 separate univariate cases**: Random Forest and genuine sktime Time Series Forest for every feature. Each case uses only that feature's 288 time samples. Other physical features and metadata labels never enter its predictors. The model target is binary `event_type` across all retained samples, including SEP rows with unknown peak flux. The energy cuts apply to the descriptive figures. Compact PNG/PDF outputs include:

- A two-model feature-by-metric heatmap: TSS, HSS, F1, precision, recall, ROC AUC, average precision and balanced accuracy.
- ROC and precision–recall curves for all ten features, grouped by model.
- Full metric CSVs, also including PR AUC, confusion counts, thresholds and undefined-metric reasons.
- Saved validation/test probabilities, the shared split table, `split_counts.csv`, `split_exclusion_counts.csv`, and `evaluation_summary.md`. The counts reconcile the dataset's full SEP population with the smaller held-out test count.

Defaults are chronological: training through 2014, validation in 2015–2019, and testing from 2020 onward. Thirty-hour boundary buffers cover the 24-hour input window plus six-hour label horizon. Known shared or boundary-straddling SEP episodes are purged. The same split is used for every feature/model. Thresholds maximize **validation TSS only**; fixed-0.5 test results are also supplied. Test samples are never oversampled or used to tune a threshold.

Instrument-calibration fitting and held-out observation days are excluded from **all classifier partitions**. Any sample whose input/label interval touches those days, plus members of the same known positive SEP episode, is recorded in `models/calibration_embargo.csv`. The run stops before fitting if a partition lacks SEP or NSEP examples after exclusions. This evaluates a retrospectively calibrated release: bridge coefficients learned from 2024/2025 instrument observations may postdate earlier test inputs, so the results do not reproduce an operational forecast using only calibration information available at each historical date.

Scores are calculated when the models run; no performance is asserted in advance. They assess discrimination on the held-out years and retained population. They do not establish physical calibration or independent SEP episodes. Overlapping windows within a partition, class imbalance and source-era changes remain relevant when interpreting usefulness.

Keep the ten CSVs together with `quality/`, the dataset manifest, row mapping, exclusions, cohort/cleaning reports, flare audits and historical-mask/calibration provenance. Keep the enhanced source archive and its native proton audits. See **[docs/REPAIRED_RELEASE.md](docs/REPAIRED_RELEASE.md)** for methods, output definitions and recovery.

Older workflows remain available for reproducibility: [strict collection](docs/COMPATIBILITY.md), [enhanced archive](docs/ENHANCED_DATASET.md), and [earlier cleaned derivatives](docs/CLEANED_DATASET.md). Those earlier filled outputs do not replace the repaired release described above.

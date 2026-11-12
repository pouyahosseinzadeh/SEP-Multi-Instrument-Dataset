# SEP dataset: repaired ten-feature release and model evaluation

This workflow reuses the completed archive, restores historical raw missingness, applies only validated source-specific proton response estimates, and removes duplicate or conflicting physical-flare identities. It produces the original ten feature files, paper figures, and separate Random Forest and Time Series Forest evaluations for each feature.

## Update the code and reuse your existing data

Download the [latest project ZIP](https://github.com/pouyahosseinzadeh/SEP-Multi-Instrument-Dataset/archive/refs/heads/main.zip). Replace `sep_dataset/`, `scripts/`, `calibration/`, `requirements.txt` and `requirements-benchmark.txt` with the updated versions. Keep your existing **`data/` and `.venv/`** folders. Copy your separately supplied `sep-historical-availability.zip` to the server project folder and extract it there. It adds two historical availability files under `calibration/`.

The input is **`data/processed/enhanced_1998_2025/`**, the completed unfiltered archive. Keep its ten original feature CSVs, `quality/` sidecars, dataset manifest and **`continuation_audit/native_sgps_*.csv`** files. The two magnetic feature CSVs are not required. Older filtered cleaned folders cannot restore excluded samples or replace the native proton audits.

From your server project folder, run:

```bash
cd ~/Pouya/Projects/SEP_DATASET_Paper/SEP-Multi-Instrument-Dataset

bash scripts/setup_env.sh
bash scripts/run_repaired_project.sh --n-jobs 4
```

Setup installs dependencies inside `.venv`, including the reference sktime Time Series Forest implementation. The runner performs **repair → validation → figures and models**, stopping at the first failed step. It does not repeat the bulk dataset collection. Packed historical availability masks recover the original missing-value locations without downloading the historical feature files again.

Final datasets are written to **`data/processed/repaired_ten_1998_2025/`**. Figures and model results are written under **`reports/figures_repaired_ten_1998_2025/`**. Source files and older outputs remain separate. The retained sample count is determined at runtime; do not reuse an earlier release's count.

## Run the stages separately

Run these commands in order, continuing after each succeeds:

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

Historical originally missing cells are restored before interpolation. A row with no valid anchors in any selected feature is excluded from all ten files, including inherited wholly filled templates whose raw rows were unavailable. Partial observational gaps receive interpolation within their own row and nearest-value edge extension, with `I`/`E` flags. Existing valid anchors and retained labels are preserved. Entirely median-filled source rows are unavailable unless the source adapter replaces them with supported observations.

After the March 2020 transition, P4/P5/P6 are recomputed from native SGPS spectra using **passed, source-specific legacy-instrument-equivalent response estimates**. The reference is the original GOES-15 W-labelled **uncorrected** channel response, including its background. These are estimated continuations of that instrument quantity; they are not native EPEAD measurements or proof of uniform physical calibration across every instrument/year.

The response models use simultaneous observations and separate held-out quiet/event tests. They do not force averages from different decades to agree. Applicability depends on satellite, product, look direction, numerical energy grid and tested spectral range. Bridges identify their modeled validation target and inherit their reference model's limitations. **Unsupported or out-of-domain windows are quarantined**, rather than filled from uncalibrated modern values. Read `proton_calibration_report.json` for actual coverage. No fixed sample count or complete coverage of every year is promised.

Physical-flare identity is checked before the cohort is selected. Conflicting SEP/NSEP labels for one identified flare quarantine the entire group; same-label duplicates keep one deterministic representative. Exact UTC onset is the primary key. Nearby onsets require additional matching region and flare-timing evidence. The final files have one canonical flare identity per row and no identity shared across SEP and NSEP. Matching evidence and exclusions are recorded.

## Figures and model results

Profiles and PCA retain five integral energy groups: **>5, >10, >30, >60 and >100 MeV**. These are metadata-based selections, not the differential P4/P5/P6 predictor names. The >5 group uses `event_type == 1` and `sep_peak_1 >= 5` pfu; the original four cuts remain unchanged. Saved `event_type == 0` samples provide the NSEP controls.

Descriptive figures include profiles, full-window PCA, Pearson/Spearman correlations, distributions, missingness and measurement-status diagnostics. Images have concise labels; statistical/source methods remain in tables and manifests.

The model stage runs **20 separate univariate cases**: Random Forest and genuine sktime Time Series Forest for every feature. Each case uses only that feature's 288 time samples. Other physical features and metadata labels never enter its predictors. Compact PNG/PDF outputs include:

- A two-model feature-by-metric heatmap: TSS, HSS, F1, precision, recall, ROC AUC, average precision and balanced accuracy.
- ROC and precision–recall curves for all ten features, grouped by model.
- Full metric CSVs, also including PR AUC, confusion counts, thresholds and undefined-metric reasons.
- Saved validation/test probabilities, the shared split table and `evaluation_summary.md` describing the measured results.

Defaults are chronological: training through 2014, validation in 2015–2019, and testing from 2020 onward. Thirty-hour boundary buffers cover the 24-hour input window plus six-hour label horizon. Known shared or boundary-straddling SEP episodes are purged. The same split is used for every feature/model. Thresholds maximize **validation TSS only**; fixed-0.5 test results are also supplied. Test samples are never oversampled or used to tune a threshold.

Instrument-calibration fitting and held-out observation days are excluded from **all classifier partitions**. Any sample whose input/label interval touches those days, plus members of the same known positive SEP episode, is recorded in `models/calibration_embargo.csv`. The run stops before fitting if a partition lacks SEP or NSEP examples after exclusions. This evaluates a retrospectively calibrated release: bridge coefficients learned from 2024/2025 instrument observations may postdate earlier test inputs, so the results do not reproduce an operational forecast using only calibration information available at each historical date.

Scores are calculated when the models run; no performance is asserted in advance. They assess discrimination on the held-out years and retained population. They do not establish physical calibration or independent SEP episodes. Overlapping windows within a partition, class imbalance and source-era changes remain relevant when interpreting usefulness.

Keep the ten CSVs together with `quality/`, the dataset manifest, row mapping, exclusions, cohort/cleaning reports, flare audits and historical-mask/calibration provenance. Keep the enhanced source archive and its native proton audits. See **[docs/REPAIRED_RELEASE.md](docs/REPAIRED_RELEASE.md)** for methods, output definitions and recovery.

Older workflows remain available for reproducibility: [strict collection](docs/COMPATIBILITY.md), [enhanced archive](docs/ENHANCED_DATASET.md), and [earlier cleaned derivatives](docs/CLEANED_DATASET.md). Those earlier filled outputs do not replace the repaired release described above.

# Repaired ten-feature release

## Required existing inputs

Repair starts from the completed **unfiltered** `data/processed/enhanced_1998_2025` archive. It reads the original ten feature CSVs, their `quality/<feature>_quality.csv` files and manifest, and the modern proton **`continuation_audit/native_sgps_*.csv`** records. The two magnetic CSVs are not required. An already filtered cleaned folder cannot recover previously excluded samples and is not accepted as the source.

Repair uses local inputs. The repository supplies `calibration/proton_calibration_bank.json`. Supply `calibration/historical_raw_masks.json.gz` and `calibration/historical_raw_masks.json.manifest.json` separately. The historical bundle records original availability and anchor fingerprints, which must match the unfiltered historical cohort. Repair copies the input hashes and provenance into the output.

Missing or inconsistent native proton audit files stop repair. Restore them from the completed enhanced archive. Already filled P4/P5/P6 cells do not substitute for missing native spectra. Targeted proton retrieval may be necessary if the audits were deleted; do not repeat all ten feature downloads by default.

## Commands

Replace the project's `sep_dataset/`, `scripts/`, `calibration/` and requirements files with the current versions, keeping `data/` and `.venv/`. Upload the separately supplied `sep-historical-availability.zip` to the server project folder. From that folder:

```bash
cd ~/Pouya/Projects/SEP_DATASET_Paper/SEP-Multi-Instrument-Dataset
unzip -o sep-historical-availability.zip
bash scripts/setup_env.sh
bash scripts/run_repaired_project.sh --n-jobs 4
```

To generate the two availability files locally from the ten original unfilled CSVs, run `.venv/bin/python scripts/build_historical_masks.py --raw-dir /path/to/originals`. The runner executes repair, full validation, descriptive figures and models in order. It stops on failure. The explicit equivalent is:

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

Use the explicit repaired input path in later commands. Dataset files are written to `data/processed/repaired_ten_1998_2025`; model results are in `reports/figures_repaired_ten_1998_2025/models/`.

## Historical availability, exclusions and filling

The published filled baseline concealed some entirely unavailable original rows behind repeated templates. The packed raw masks restore original missing positions for all ten features, while checking known observed anchors at original source precision. Missing codes include `-99999`, `-9999`, nonfinite values and documented OMNI positive fill codes. Valid zeros and signed Vx are preserved.

A wholly unavailable selected feature excludes that sample from all ten files. An inherited all-median source row also lacks observed anchors. Partial observational gaps receive linear interpolation between valid values and nearest-value edge extension in the same row. No class median or another sample invents a whole window. Filled cells are explicitly `I`/`E`; existing source provenance remains available.

Consequently, the final population changes. Do not quote an earlier release's 18,417 or 33,650 samples as the new count. Read `cohort_counts.csv`, `exclusions.csv` and `availability_report.txt` after the run.

## Proton definition and calibration scope

The reference quantity is the original GOES-15 W-labelled **uncorrected EPS/EPEAD differential channel response**, including its background. Modern P4/P5/P6 are estimates of that instrument quantity derived from native SGPS spectra. They are not recovered native EPEAD observations, background-free solar proton fluxes, or proof of uniform physical calibration across every mission and year.

The base model fits nonnegative spectral coefficients and a nonnegative intercept to simultaneous September 2017 observations. Separate held-out quiet and event intervals test coverage, bias, error and event variation. This is not population-mean matching across decades.

A bridge can map another supported product to an already validated reference estimate. Its held-out target is modeled, rather than an independent legacy measurement. The bank records that distinction, reference lineage, fitting/validation periods, gate results, source keys and tested spectral envelope. Such a bridge inherits the reference model's error; a passed bridge gate is not independent evidence of physical equivalence to the retired detector.

The published bank has three passed modes: GOES-16 SGPS L2, GOES-16 SGPS L1b and GOES-18 SGPS L2. Applicability is specific to satellite, product, detector/look direction and energy-coordinate basis, with numerical grid and spectral-range checks. Native valid zero fluxes are retained in response estimation. Unsupported sources, energy grids or out-of-domain native points quarantine the affected window. They are not turned into ordinary gaps and interpolated across a row. Supported observational gaps can receive the documented row-wise filling.

The output `proton_calibration_report.json` and copied `calibration_bank.json` identify the modes actually used. Unsupported windows are counted. The release does not promise that every 2020–2025 sample or future product is covered, and no universal correction factor is asserted.

## SEP/NSEP physical-flare separation

Exact UTC onset is the primary identity key. Nearby onsets require the same known NOAA region and an exactly shared auxiliary start, peak or end timestamp. Proximity, overlapping flare intervals or a shared SEP episode alone cannot identify two flares as identical.

Conflicting SEP/NSEP groups and ambiguous region identities are quarantined. Same-label duplicates keep one deterministic representative, preserving its original metadata and label. The final release has one canonical identified flare per row, with no identity shared between classes. Incomplete catalog metadata can still limit physical identification; the audit records the exact matching evidence.

Canonical IDs and original sample/source-row IDs live in the sidecars and row mapping, preserving the original 52 metadata columns. `flare_identity_audit.csv` covers every source row, and `exclusions.csv` combines identity and unavailable-feature reasons.

## Univariate model evaluation

Each feature is evaluated separately using 288 chronological pre-flare values. Random Forest consumes those lag values; the genuine sktime Time Series Forest builds interval-statistic trees using means, standard deviations and slopes. Twenty cases are evaluated, rather than combining ten physical features into one model.

To run models alone:

```bash
.venv/bin/python -u -m sep_dataset.cli benchmark \
  --dataset-dir data/processed/repaired_ten_1998_2025 \
  --outdir reports/figures_repaired_ten_1998_2025/models \
  --n-jobs 4
```

The defaults are 200 trees, training through 2014, validation in 2015–2019, and testing from 2020 onward. Thirty-hour boundary buffers cover input-window and label-horizon overlap. Known shared or straddling SEP episodes are purged. All cases use the same saved identities and split.

The copied calibration bank records the actual fitting and held-out observation days for every instrument-response artifact. The union of those days is embargoed from classifier training, validation and testing: a sample is excluded if its interval from 24 hours before flare onset through six hours after onset intersects a calibration day. All members of a known positive SEP episode intersecting those days are also excluded. The saved split and `calibration_embargo.csv` identify each affected sample, its reason and calibration days. A missing calibration-date audit is an error, and a partition losing either class stops the benchmark before model fitting; dates are not adjusted in response to test labels.

The threshold is selected using validation TSS only, with deterministic tie-breaking; fixed-0.5 results are also exported. Test labels do not tune thresholds or sampling. Random Forest uses balanced class weights; the reference TSF uses documented unweighted fitting. No test oversampling is performed. Undefined metrics, such as precision when there are no positive predictions, remain unavailable with reasons.

Outputs include:

- `univariate_test_metrics.csv`: TSS, HSS, F1, precision, recall, ROC AUC, PR AUC, average precision, balanced accuracy, confusion counts and additional diagnostics.
- `univariate_test_metrics_fixed_0_5.csv` and `validation_thresholds.csv`.
- `univariate_metrics.png/.pdf`, `univariate_roc.png/.pdf`, and `univariate_pr.png/.pdf`.
- `cases/<feature>/<model>/validation_predictions.csv` and `test_predictions.csv`.
- `splits.csv`, `calibration_embargo.csv`, `source_era_counts.csv`, `benchmark_manifest.json`, and `evaluation_summary.md`.

The written analysis summarizes measured held-out results and no-skill references, including SEP prevalence for average precision. No favourable score is promised before the run. These scores assess discrimination in the stated population; they do not validate instrument physics or source labels. Dependence and overlapping windows within partitions, imbalance and source-era changes remain relevant. No independent-sample confidence intervals are claimed.

This is evaluation of retrospectively calibrated measurements, rather than a strict operational forecast replay. Some response bridges use 2024/2025 instrument observations and therefore postdate earlier classifier test inputs. Embargoing the exact calibration-development days prevents those observations from entering the classifier holdout, but does not recreate the instrument-response knowledge available at every historical forecast date. Model discrimination does not independently validate the physical calibration.

## Integrity and recovery

The workflow name is `repaired_ten_feature_release_v2`. Validation checks the ten 340-column files, common metadata/order/labels, preserved adapted anchors, finite values, quality-mask counts, source/retained/excluded accounting and hashes. Repaired-specific checks cover flare identity, historical availability and the passed calibration/provenance bank. Completion is recorded only after validation passes.

Keep the ten CSVs with `quality/`, dataset manifest, row mapping, exclusions, cohort/cleaning reports, flare audits and historical-mask/calibration provenance. Native measurement audits remain in the enhanced archive; retain them for source verification.

Repair writes one derivative without creating a duplicate staging dataset. Its disk preflight estimates remaining allocation, but a shared filesystem can change during a run. If storage is insufficient, free space or choose another mounted output disk, then repeat the same command. Completed feature checkpoints are verified and reused. Do not remove the enhanced input or native proton audits to make room.

Model evaluation saves predictions and checksummed records instead of large trained-model dumps. Matching inputs and settings allow completed cases to resume. Changed data or policies require new computation. The manuscript must still describe modeled continuations, source limits, selection and filling; software integrity checks are not a replacement for Technical Validation.

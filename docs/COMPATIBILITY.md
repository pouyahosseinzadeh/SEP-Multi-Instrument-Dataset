# Historical preservation and extension methodology

## Preserved dataset

The frozen historical files come from [Zenodo record 17595580](https://zenodo.org/records/17595580), or from the user's `--baseline-dir`. Their 17,794 rows comprise 252 SEP and 17,542 NSEP samples. All metadata, labels, row order and 288-point feature blocks in the chosen baseline remain unchanged. The published extra `Label` column is dropped; `event_type` and the original 52-column metadata schema remain. CSV formatting may change; validation checks historical numerical values and metadata against fingerprints. Published CSV serialization differs at floating-point rounding precision from some local originals; `--baseline-dir` preserves those specific local values exactly.

| Prefix | Measurement |
|---|---|
| `Vx` | Solar-wind X velocity |
| `flow_speed` | Solar-wind speed |
| `proton_density` | Solar-wind proton density |
| `T` | Proton temperature |
| `F` | IMF magnitude |
| `Xs`, `Xl` | Short and long GOES X-ray channels |
| `P4`, `P5`, `P6` | Original nominal differential proton channels |

All ten outputs share identical metadata and sample order. The first 17,794 rows are the frozen baseline; new rows are appended chronologically. The published `Label` is an energy-category display field (`NSEP`, `gt10`, `gt30`, `gt60`, `gt100`), not the binary target: some original positive events below its threshold carry `NSEP` in that display field. The authoritative `event_type` is never derived from `Label`. Existing baseline interpolation masks cannot be reconstructed from filled CSVs and are explicitly marked unknown.

## Flare cohort and observed labels

The extension uses observed SWPC C/M/X flares retrieved through HEK, beginning January 1, 2014 by default. It retains a flare cohort rather than generating random quiet-time controls. Historical labels are never recomputed.

The observational definition follows [MEMPSEP-III, sections 3–4](https://arxiv.org/pdf/2310.15390). Corrected integral >10 MeV proton flux must exceed both 5 pfu and the maximum three-hour pre-flare background for three consecutive five-minute samples within six hours after flare onset. The default requires complete observed background and search-window coverage. Missing flux produces an unresolved label, not an NSEP sample. Exclusions and coverage remain in `label_audit.csv`.

The authors' precise competing-flare grouping implementation is not published. This extension therefore uses an explicit deterministic rule: in the same above-5-pfu episode, eligible flares preceding its first enhancement by at most six hours compete; the largest X-ray flare wins, with earlier onset breaking equal-class ties. The rule identifier and winner are audited. This is a documented extension, not a claim to reproduce unavailable original labeling code exactly. Event end requires three consecutive samples below 5 pfu and uninterrupted preceding observations.

Only adequately covered observed metrics are populated. Unavailable metadata remain `NaN`; unrelated events do not supply replacement values. In particular, the original `sep_fluence_*` normalization is insufficiently documented, so these fields remain `NaN` for new rows. Separately named pfu-second integrals appear in the audit when coverage supports them. They are not inserted into fields with unknown units or normalization.

## Instrument identity and calibration

OMNI supplies the five solar-wind/IMF features; archived operational GOES XRS supplies `Xs`/`Xl`; native GOES-15 EPEAD supplies the nominal legacy proton channels during their available extension interval. Feature source manifests preserve actual URLs, parameters, cadence, timing and calibration-era notes.

**P4/P5/P6 are unavailable after March 4, 2020, 23:55 UTC.** GOES-R SGPS differential energy responses differ from the original channels. They are not renamed P4/P5/P6, and integral >10/>30/>60 MeV flux is not substituted for differential channels. Affected cells remain `-9999`, including in `_filled.csv`, and appear in `unavailable_channels.csv`. Boundary-crossing windows can contain both available and structurally unavailable points.

Nominal names do not imply identical satellite energy response or calibration. GOES-15 EPEAD differs from older EPS instruments. GOES-8–15 operational XRS scaling differs from GOES-R calibration; native archived values are retained with explicit era provenance rather than an assumed rescaling. The project does not assert cross-era instrument homogeneity.

## Windows, interpolation and causality

Each feature has 288 five-minute bins. `FlrOnset` is rounded up to the next five-minute boundary; the grid spans aligned onset minus 24 hours through aligned onset minus five minutes. Columns run `<feature>_288` through `<feature>_1`. Historical feature blocks remain as published.

For new rows, documented source fill values are cleaned, interior gaps are interpolated linearly and edges use the nearest observed value within a covered segment. Wholly missing covered segments use frozen historical per-time-column medians, with a historical global median as the final fallback. Those are imputed values, not observations. Structural absence splits segments and is never interpolated, edge-filled or median-filled. Remaining missing cells are written as `-9999`.

Some native timestamps precede the end of their measurement interval. Source-specific lookahead guards remove measurements whose integration interval would extend beyond flare onset. XRS uses trailing aggregation. Exclusions and subsequent filling are recorded separately. Historical causality is not retrospectively asserted when original timing provenance is unknown.

`quality/<feature>_quality.csv` includes row-level filling counts, source era/method, `unavailable_channel`, and two 288-character masks in time-column order:

| Mask | Codes |
|---|---|
| `point_status` | `O` observed; `I` linear interpolation; `E` edge fill; `M` historical-median fill; `S` structural absence; `G` remaining gap; `U` unknown historical provenance |
| `causality_mask` | `1` excluded by timing guard; `0` not excluded; `U` unknown historical provenance |

After-filling completeness is not original observational coverage. Prefill summaries concern audited extension rows; historical prefill quality remains unknown. Point masks support observed-only and low-imputation sensitivity analyses without altering the preserved historical values.

## Reproducibility reports

- `dataset_manifest.json`: completed files, historical fingerprints, checksums and source/calibration records.
- `build_config.json`, `build_status.json`, `label_checkpoint.json`: configuration, progress and reusable checkpoint state.
- `flare_cohort.csv`, `extension_samples.csv`, `label_audit.csv`: candidate flares, resolved rows and label/coverage decisions.
- `quality_summary.csv`, `quality/<feature>_quality.csv`: aggregate and per-point filling provenance.
- `unavailable_channels.csv`, `unavailable_metadata.csv`, `availability_report.txt`: explicit unavailable channels and fields.

Downloads and compatible completed feature checkpoints are reused on rerun. Validation checks ten schemas, identical metadata/order, historical preservation, checksums, masks and structural absence. A completed build may include reported unavailability; it does not simulate measurements to hide those limitations.

## Figure compatibility

The original comparisons use `event_type == 1` and `sep_peak_2 >= 10` for >10 MeV. The >30, >60 and >100 MeV comparisons use `sep_peak_3`, `sep_peak_4`, `sep_peak_5 >= 5` respectively. Controls are all `event_type == 0` flare rows. The energy inequality denotes the integral channel; its peak threshold is in pfu.

Means, normal-approximation pointwise 95% confidence intervals and Welch tests follow the original notebook. Additional BH q-values are explicitly corrected within each feature/band's time-point tests. PCA uses all 288 columns standardized jointly over selected SEP/NSEP rows, matching the original per-feature method. These are descriptive fits, not held-out validation. Optional t-SNE records sampling and its seed.

Entirely unavailable windows are omitted from means and PCA and counted in the reports. Score files retain sample identifiers and provenance when supplied. Yearly feature availability exposes gaps. Imputation, calibration era, class imbalance, temporal dependence and association uncertainty require sensitivity analysis before scientific claims.

Three old plotting defects are corrected without changing baseline values: density now uses the filled file instead of a sentinel-contaminated unfilled file; valid temperatures near 100,000 K are not removed as a generic fill flag; the last time bin is displayed at five minutes before aligned onset instead of zero. The original `linspace(24, 0, 288)` coordinate remains in summary tables for comparison. The historical heatmap filename remains, but its label correctly describes a time-normalized mean contrast rather than Cohen's d.

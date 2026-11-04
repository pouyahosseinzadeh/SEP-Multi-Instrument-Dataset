# Enhanced twelve-feature dataset and publication diagnostics

This document describes the twelve-feature source archive before the final offline cleanup. Default `build` and `enhance` now also create the final filled, aligned release in `data/processed/cleaned_1998_2025`. Default validation and plotting use that final release. If the twelve source files are already complete, run `python -m sep_dataset.cli clean`; see [CLEANED_DATASET.md](CLEANED_DATASET.md) for interpolation, source-fill masking, shared row exclusions and the exact run commands. The source policies and unavailable-value descriptions below continue to apply to the archived inputs.

## Coverage and intended interpretation

The enhanced release retains the original flare cohort and 340-column per-feature layout, adds `USFLUXL` and `R_VALUE`, and supplies an explicitly derived modern continuation for unavailable proton measurements. It does not claim complete native coverage for every feature, or uniform calibration across instruments. Missing active-region identifiers, ambiguous patches, rejected quality flags, stale records, unsupported dates and invalid energy brackets can leave `-9999` cells. Read the availability and provenance reports before scientific interpretation.

`build` first creates or reuses the strict ten-file dataset under `data/processed/compatible_1998_2025`, then writes twelve enhanced files to `data/processed/enhanced_1998_2025`. The strict source files are retained unchanged. Existing complete runs can use `enhance --legacy-dir ...` without repeating the original data collection. The historical labels, sample order and original ten 1998–2013 feature blocks are preserved; the two magnetic feature blocks are newly collected for those same samples.

Every enhanced file contains the same 52 metadata fields and exactly 288 five-minute feature columns. Rows are flare samples, with 17,794 frozen historical rows followed by the resolved extension. There are no random quiet-time replacements. The observational extension labels and unavailable legacy metadata follow the existing [compatibility method](COMPATIBILITY.md). Missing metadata and undocumented original fluence normalization are not invented.

## Run commands

From the latest downloaded project folder:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -u -m sep_dataset.cli build --end-date 2025-12-31
python -m sep_dataset.cli validate --dataset-dir data/processed/enhanced_1998_2025
MPLBACKEND=Agg python -u -m sep_dataset.cli plot \
  --dataset-dir data/processed/enhanced_1998_2025 \
  --outdir reports/figures_enhanced_1998_2025
```

To reuse completed original outputs:

```bash
python -u -m sep_dataset.cli enhance \
  --legacy-dir data/processed/compatible_1998_2025 \
  --outdir data/processed/enhanced_1998_2025
```

Then validate and plot as above. `--legacy-dir` can point to another complete compatible run. `build --legacy-only` retains the strict ten-feature workflow; its default output is the compatible folder. Do not use the same folder for strict inputs and enhanced outputs.

## Modern P4/P5/P6 continuation

Original EPEAD channels cease in March 2020. The enhanced output replaces only the corresponding reported continuation gaps with **SGPS-derived differential point-flux estimates** where observations support them:

| Output prefix | Target energy |
|---|---:|
| P4 | 30.6 MeV |
| P5 | 63.1 MeV |
| P6 | 165 MeV |

The estimate interpolates linearly in log(flux) against log(energy) between adjacent positive, quality-valid native energy channels. It does not extrapolate beyond native energy coverage and does not replace differential channels with integral proton flux. Units are differential protons/(cm² s sr MeV); per-keV source values are converted explicitly when needed.

Native L2 five-minute SGPS products are used where available. A native L1b one-second-to-five-minute bridge covers available earlier 2020 products; it lacks the L2 temperature correction and uses nominal energy-label bounds where calibrated effective energies are unavailable. A separately reprocessed September 2017 archive is also supported. Product era, energy-coordinate basis, sensor/yaw selection, quality/advisory information, bracket fluxes, uncertainty and actual source URLs are retained in source records and cached native data.

Physically westward sensor selection uses the spacecraft yaw information. Left-labelled integrations receive the appropriate timing guard, so a bin extending past flare onset is excluded. A bounded backward hold can supply a recent valid derived value for up to ten minutes; it is flagged `H`, not a new observation. Energy-interpolated points are flagged `D`. Invalid brackets, missing products or stale observations remain missing.

**Energy matching does not prove equality to the legacy EPS/EPEAD detector response, background or calibration.** The continuation does not resurrect native EPEAD measurements or apply an asserted universal bias correction. Native L1b and L2 processing differences remain explicit. The original strict files remain available for analyses requiring native legacy channels only.

## Magnetic features

`USFLUXL` is the native active-region line-of-sight unsigned photospheric flux in Mx. `R_VALUE` is the native logarithmic Schrijver R index; it is retained as supplied and is not exponentiated or logged again.

Sources are native JSOC `mdi.smarp_cea_96m` and `hmi.sharp_cea_720s` keyword records: approximately 96-minute MDI and 12-minute HMI cadence, with the HMI era beginning May 2010. The release does not substitute a regressed/reconstructed magnetic table. Native MDI/HMI calibration differences are retained; no scaling or cross-instrument equivalence is assumed.

JSOC can omit its keyword arrays for a successful query with zero records (`status: 0, count: 0`). These explicit empty results are retained as source gaps, including in cached responses, and do not stop collection. The affected magnetic values remain missing and are audited; no values are invented. Malformed responses, nonzero error statuses and inconsistent record counts still raise errors. After updating the collector, rerun `enhance` to reuse the completed feature checkpoints and source cache.

Records must match the sample's NOAA active-region membership in a unique native magnetic patch. A patch containing multiple NOAA regions is retained only when it contains the requested region, and is explicitly flagged through `multi_region_patch_points` and `matched_noaa_membership`; the parameter describes that whole patch, not an isolated subregion. Simultaneously conflicting patches, missing identifiers and failed quality cuts remain unavailable. Four-digit historical NOAA identifiers after the numbering rollover are expanded only when the full identifier is confirmed in native records; original metadata are unchanged and normalization is audited. Native quality cuts are `0 <= QUALITY < 262144` for SMARP and `0 <= QUALITY < 65536` for SHARP. Accepted nonzero quality values remain in native audit records.

Native TAI timestamps are converted to UTC using Astropy. Sampling uses the later of `T_REC` and `T_OBS`, plus integration guards (five minutes for MDI; twelve for HMI), followed by backward sampling. A value is held for at most two native cadences. Newly available native samples are `O`; repeated five-minute values are `H`. The held grid does not create 288 independent magnetograms. There is no linear interpolation or median filling of missing magnetic records.

These definitive products are retrospective. Timing guards prevent use of measurement integrations extending into the future, but do not reconstruct the original real-time publication latency. Operational forecasting claims require a separate latency-aware evaluation.

### Optional native keyword CSV

If supplying an offline export, add `--magnetic-csv /path/to/native_keywords.csv` to `build` or `enhance`. One CSV can contain records from both native series. Required columns are:

| Column | Required content |
|---|---|
| `series` | `mdi.smarp_cea_96m` or `hmi.sharp_cea_720s` |
| `T_REC`, `T_OBS` | Explicit native `YYYY.MM.DD_HH:MM:SS_TAI`, or UTC/offset timestamps |
| `TARPNUM` or `HARPNUM` | Native patch number for the corresponding series |
| `NOAA_AR`, `NOAA_ARS` | Primary NOAA identifier and native membership list |
| `QUALITY` | Original native quality value, including supported hexadecimal notation |
| `USFLUXL`, `R_VALUE` | Original native parameter values |

Bare timezone-free timestamps are not silently treated as UTC. Do not pass an already sampled or median-filled 288-column dataset as the native export. Parameter values, timestamps, region association and quality information must retain their native meanings. Source failures raise actionable errors; an entirely empty new magnetic feature is rejected.

A native keyword export helper is available when source access works on another machine:

```bash
python -m sep_dataset.magnetic_sources \
  --series hmi.sharp_cea_720s \
  --start 2025-01-01 --end 2025-02-01 \
  --output SHARP_2025-01.csv
```

The export interval has an exclusive end in UTC. The corresponding MDI series is `mdi.smarp_cea_96m`. Retain the complete native schema when combining exports into the CSV supplied to `--magnetic-csv`.

## Quality masks and data reports

`continuation_audit/` contains per-query-month SGPS native fluxes (including valid zeros), energies and quality values, plus each predictor's interpolation brackets, uncertainty and availability timestamps. These files are checksum-protected and repaired on resume if missing or modified. Query windows follow the month of the flare anchors and can include the preceding day. All twelve row-quality tables use the same sample identifiers.

`quality/<feature>_quality.csv` retains a 288-character `point_status` mask in the same order as `<feature>_288 ... <feature>_1`:

| Code | Meaning |
|---|---|
| O | Native observation |
| D | SGPS-derived point-flux estimate |
| H | Bounded causal hold of an observation or estimate |
| I | Interior linear filling in the original workflow |
| E | Original nearest-edge filling |
| M | Original historical-median fallback |
| S | Structural absence |
| G | Remaining unavailable point; detailed native reasons remain in reports |
| U | Unknown historical provenance in frozen filled data |

Thus `_filled.csv` does not mean native observations at every point. The enhanced release does not silently median-fill unsupported modern proton or magnetic measurements. The original workflow's documented interpolation and median fallback remain attached to its own provenance.

`dataset_manifest.json` records files/checksums, source metadata, processing methods and calibration caveats. `quality_summary.csv`, per-feature quality sidecars, `unavailable_channels.csv` and `availability_report.txt` summarize limitations. Original cohort/label audits remain associated with the strict input build. Validation checks common metadata/order, twelve feature blocks, quality masks and unchanged protected historical data.

## Expanded figures and statistical units

The normal `plot` command retains the original per-feature mean profiles, uncertainty intervals, energy thresholds, Welch tests and **full 288-point PCA**. Its additional `paper/` folder contains twelve diagnostic figures, each in PNG/PDF, with numerical CSVs:

- Pearson and Spearman feature correlations, including SEP/NSEP comparisons and calendar/source-era sensitivity.
- Raw versus flux-log Pearson correlations and observed-only versus saved-value comparisons.
- SEP/NSEP window-mean distributions and within-window variability.
- Native-only versus filled/derived/held window-mean sensitivity.
- Before/after missingness, yearly coverage, and native/derived/held processing fractions.
- Time-resolved feature/label associations, with valid sample counts at each time offset.

For feature correlations, **one flare window is one row**. Means and medians are computed per sample before correlation; five-minute cells are not pooled as independent observations. `feature_correlations.csv` includes `n_pairwise` for each Pearson/Spearman coefficient, policy, aggregation, transform and group. Small or numerically constant cohorts receive `NaN` rather than a coefficient. `correlations_by_instrument_era.csv` uses actual per-feature source-era strings; calendar periods are additional descriptive partitions, not proof of instrument equivalence.

The flux-log variant uses log10 of positive window summaries for Xs, Xl, P4, P5, P6 and USFLUXL. Signed Vx is unchanged; R_VALUE is already logarithmic and remains unchanged. Nonpositive exclusions are counted in `transform_sample_counts.csv`. A monotonic log preserves Spearman ranks for a fixed positive cohort, so differences can arise from excluded nonpositive summaries. Time-offset association plots log positive flux points before correlating with labels; this is separately documented from log-transforming a window summary.

The default saved-value view includes finite historical and enhanced values, with provenance reported. The **native-only sensitivity uses finite O-coded points only**, requiring at least three native observations per window; I/E/M/S/G/U/D/H points do not enter its window summaries. Frozen historical U points cannot be certified as observed-only. Pairwise sample counts can therefore differ sharply between views and instrument eras. Before-gap-fill coverage is reported only where provenance is known, with derived/held data distinguished from native observations.

No extra values are imputed by plotting. The figures report descriptive associations without significance or forecasting-performance claims. Overlapping windows, temporal dependence, repeated flare associations, class imbalance, imputation, native cadence and calibration changes require separate sensitivity studies. `paper_figures_manifest.json` records the exact policies, transforms, thresholds and limitations.

To run only the additional diagnostics:

```bash
MPLBACKEND=Agg python -m sep_dataset.cli paper-plot \
  --dataset-dir data/processed/enhanced_1998_2025 \
  --outdir reports/figures_enhanced_1998_2025/paper
```

Use `plot --original-figures-only` to omit the additional diagnostics or `plot --tsne` to request the optional slower t-SNE views.

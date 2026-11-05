# Final filled and aligned feature releases

The offline cleaner fills partially available rows using values within that row. It removes a sample from every output feature file when any selected feature row has no usable value after source-fill masking. The same exclusion mask preserves sample identities, ordering, metadata and SEP/NSEP labels across the selected files.

## Create the final ten-feature release from existing downloads

Use the completed, unfiltered source archive in `data/processed/enhanced_1998_2025`. Update the code while keeping your existing `data` and `.venv` folders. From the server project folder, run:

```bash
cd ~/Pouya/Projects/SEP_DATASET_Paper/SEP-Multi-Instrument-Dataset
```

First create the ten-feature output:

```bash
.venv/bin/python -u -m sep_dataset.cli clean --original-ten-only \
  --dataset-dir data/processed/enhanced_1998_2025 \
  --outdir data/processed/cleaned_ten_1998_2025
```

Then validate it:

```bash
.venv/bin/python -m sep_dataset.cli validate \
  --dataset-dir data/processed/cleaned_ten_1998_2025
```

Finally regenerate figures from those files:

```bash
.venv/bin/python -u -m sep_dataset.cli plot \
  --dataset-dir data/processed/cleaned_ten_1998_2025 \
  --outdir reports/figures_final_ten_1998_2025
```

The new release contains **Vx, flow_speed, proton_density, T, F, Xs, Xl, P4, P5 and P6**. USFLUXL and R_VALUE do not participate in its exclusion rule, validation or plots. Their source files are not required, even when the enhanced archive manifest lists twelve features. The retained sample count is calculated from the selected ten inputs and recorded in `cohort_counts.csv`; no fixed count is promised in advance.

These commands perform no source download and do not change the source archive or older outputs. `clean --original-ten-only` selects `data/processed/cleaned_ten_1998_2025` by default. Explicit input/output paths are recommended so each release is unambiguous. Use a new output directory rather than mixing ten- and twelve-feature releases.

An earlier twelve-feature cleanup excluded samples based on magnetic coverage as well as the original ten features. Plotting only ten features from that saved subset does not restore those samples. The new ten-feature cleanup recalculates the cohort from the enhanced archive. An already filtered `cleaned_1998_2025` directory is not a suitable source for recovering excluded samples; use the enhanced archive.

The validator checks schema, alignment, metadata, checksums, quality masks, retained/excluded sample accounting and final feature completeness. Source measurements, SEP label associations and the documented SGPS transition still require scientific validation. Keep the per-row quality files and source reports with the ten CSVs.

## Figures from an existing cleaned release

For later figure-only reruns, repeat the `plot` command above. It reads saved data without cleaning or rewriting it. All figures use only the original ten features. Default `plot`, `paper-plot` and `validate` still select the older `data/processed/cleaned_1998_2025` directory; supply `--dataset-dir data/processed/cleaned_ten_1998_2025` for the ten-feature release.

The >5 MeV group selects `event_type == 1` rows with `sep_peak_1 >= 5` pfu. The original >10, >30, >60 and >100 MeV selections are unchanged. NSEP controls remain all saved `event_type == 0` rows. Profiles retain shaded 95% confidence intervals. Statistical tests and source notes remain in CSVs and manifests instead of method/source footers or significance overlays on the images.

## Optional twelve-feature cleanup

The original default `clean` command remains available for compatibility:

```bash
.venv/bin/python -u -m sep_dataset.cli clean \
  --dataset-dir data/processed/enhanced_1998_2025 \
  --outdir data/processed/cleaned_1998_2025
```

It includes USFLUXL and R_VALUE in the shared exclusion rule. New `build` and `enhance` runs retain that twelve-feature cleanup behavior; `--clean-outdir` selects its destination. The separate `--original-ten-only` command creates the ten-feature cohort from the completed archive. Source and output directories must be different.

## Invalid values

Only the 288 feature columns are processed. Metadata is preserved exactly for retained samples and may still contain unavailable entries.

The cleaner treats `-9999` and nonfinite feature values as unavailable. It additionally recognizes these documented OMNI fill values and their exact float32 representations:

| Feature | Source fill value |
|---|---:|
| F | 9999.99 nT |
| Vx, flow_speed | 99999.9 km/s |
| proton_density | 999.99 cm^-3 |
| T | 9999999 K |

These follow the [NASA OMNIWeb data definitions](https://omniweb.gsfc.nasa.gov/html/sc_merge_data1.html). They are exact source-code matches, not broad clipping of large observations. Valid zero values and signed Vx measurements are retained. Full historical rows made only of a positive source-fill value are unavailable rows, even when the previous report counted them as finite.

## Filling within a retained row

The feature columns are uniformly spaced in chronological order, from `_288` to `_1`:

1. Gaps between usable values receive linear interpolation in time.
2. Leading and trailing gaps use the nearest usable value in that same row.
3. A row with exactly one usable value extends that value across its other bins and is explicitly audited as edge extension.
4. A row with no usable value is excluded. No mean/median from another sample, class or year supplies a fabricated full row.

The rule fills all remaining unavailable cells in a retained row, including long within-window gaps; it does not guarantee that those estimates reproduce unobserved physical variations. The quality sidecar records how many cells and which bins were filled. Consumers can select stricter coverage criteria using the pre-cleaning counts.

Interpolation is retrospective within the pre-flare window. It uses no measurements from after the sample's pre-flare window, but an interior bin can use a later bin in that window. New interpolated bins must not be described as causal native observations. The SGPS source estimates, calibration caveats and source quality controls remain documented in [ENHANCED_DATASET.md](ENHANCED_DATASET.md).

## Quality and selection reports

Final CSV names and the 52-metadata-plus-288-value layout match the input release. Every final feature file has the same retained samples and no missing feature values.

`quality/<feature>_quality.csv` retains stable `sample_id`, source row index, source and before-cleaning point-status masks, and new interpolation/edge-fill counts. Existing anchors preserve their `O`, `D`, `H` or historical `U` provenance. Newly interpolated cells are `I`; new edge extensions are `E`. Original historical acquisition missingness remains unknown where it was unknown before cleanup.

`exclusions.csv` identifies dropped samples and the feature or features lacking any usable value. `cohort_counts.csv` reports the sample and SEP/NSEP counts before and after selection by year. `quality_summary.csv` and `availability_report.txt` report invalid values, filled cells and final completeness. The manifest records input hashes, processing policy, source preservation and output checksums. The validator checks file structure, sample/label alignment, quality masks, checksums and absence of unavailable final values.

The common exclusion rule is calculated only over the features selected for this release. For the ten-feature mode, magnetic availability does not remove samples. The plots describe the retained cohort, not every source sample. Report the selected feature set, final counts and exclusion reasons in the paper. Existing source imputation and SGPS estimates remain flagged; software validation does not by itself establish source accuracy, label validity or cross-instrument equivalence.

## Disk space and recovery

The cleanup keeps the source archive and writes separate final files. Check `df -h .` before running when the server is nearly full. The cleaner checks available space against a source-size-based output estimate before writing the final release; this estimate is not a guarantee for an actively shared filesystem. Keep completed source outputs and caches. A storage failure requires freeing space or using a larger mounted filesystem, then rerunning `clean`; it does not require re-downloading all sources.

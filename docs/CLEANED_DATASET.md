# Final filled and aligned twelve-feature release

The final offline cleanup applies the same rule to every feature: fill partially available rows from values within that row, and exclude a sample from all twelve files if any feature row is entirely unavailable after source-fill masking. A shared exclusion mask preserves sample identities, ordering, metadata and SEP/NSEP labels across feature files.

## Existing completed downloads

From your updated project folder, run:

```bash
.venv/bin/python -u -m sep_dataset.cli clean
.venv/bin/python -m sep_dataset.cli validate
.venv/bin/python -u -m sep_dataset.cli plot
```

`clean` reads `data/processed/enhanced_1998_2025` and writes `data/processed/cleaned_1998_2025`. It performs no source download and does not require a raw-cache rebuild. New `build` and `enhance` runs invoke cleanup after completing the source archive. `--clean-outdir` selects a different final output for those commands. `clean --dataset-dir PATH --outdir PATH` selects existing source and final folders directly. Source and final folders must be different.

Default `validate`, `plot` and `paper-plot` use the cleaned output. Figures are written to `reports/figures_cleaned_1998_2025`. Explicit `--dataset-dir` still permits inspecting a source archive with gaps.

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

Interpolation is retrospective within the pre-flare window. It uses no measurements from after the sample's pre-flare window, but an interior bin can use a later bin in that window. New interpolated bins must not be described as causal native observations. The SGPS source estimates, native magnetic cadence, calibration caveats and source quality controls remain documented in [ENHANCED_DATASET.md](ENHANCED_DATASET.md).

## Quality and selection reports

Final CSV names and the 52-metadata-plus-288-value layout match the input release. Every final feature file has the same retained samples and no missing feature values.

`quality/<feature>_quality.csv` retains stable `sample_id`, source row index, source and before-cleaning point-status masks, and new interpolation/edge-fill counts. Existing anchors preserve their `O`, `D`, `H` or historical `U` provenance. Newly interpolated cells are `I`; new edge extensions are `E`. Original historical acquisition missingness remains unknown where it was unknown before cleanup.

`exclusions.csv` identifies dropped samples and the feature or features lacking any usable value. `cohort_counts.csv` reports the sample and SEP/NSEP counts before and after selection by year. `quality_summary.csv` and `availability_report.txt` report invalid values, filled cells and final completeness. The manifest records input hashes, processing policy, source preservation and output checksums. The validator checks file structure, sample/label alignment, quality masks, checksums and absence of unavailable final values.

The global exclusion rule can remove many samples if one feature has poor coverage. The resulting plots describe the retained cohort, not the full pre-cleaning population. Report that selection in the paper, and investigate suspicious source/region-matching losses separately rather than claiming that interpolation recovered an entirely missing row.

## Disk space and recovery

The cleanup keeps the source archive and writes separate final files. Check `df -h .` before running when the server is nearly full. The cleaner checks available space against a source-size-based output estimate before writing the final release; this estimate is not a guarantee for an actively shared filesystem. Keep completed source outputs and caches. A storage failure requires freeing space or using a larger mounted filesystem, then rerunning `clean`; it does not require re-downloading all sources.

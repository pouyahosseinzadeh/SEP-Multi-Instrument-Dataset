# Migration to the ten-file extension through 2025

The earlier 1998–2024 wide-dataset pipeline changed the sample construction and feature set. It is superseded by the original-layout ten-file extension.

Use the current [README](../README.md) for installation, build, validation and figure commands. See [COMPATIBILITY.md](COMPATIBILITY.md) for historical preservation, observational labels, interpolation, source calibration and explicit channel unavailability.

The supported commands are `build --end-date 2025-12-31 --outdir ...`, `validate --dataset-dir ...` and `plot --dataset-dir ...`. The earlier `--catalog-only`, `--negative-ratio`, `--memsep-csv` and `plot --dataset-csv` instructions do not apply to this workflow. Keep outputs and caches from that different construction separately; they are not the ten compatible extended feature files.

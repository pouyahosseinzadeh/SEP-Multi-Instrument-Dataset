#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
SEP_PYTHON="${SEP_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
SOURCE_DIR="data/processed/enhanced_1998_2025"
FINAL_DIR="data/processed/repaired_ten_1998_2025"
FIGURE_DIR="reports/figures_repaired_ten_1998_2025"
JOBS=1
TREES=200
TRAIN_END="2014-12-31"
VALIDATION_END="2019-12-31"
while (($#)); do
  case "$1" in
    --dataset-dir|--outdir|--figure-outdir|--n-jobs|--n-estimators|--train-end|--validation-end)
      if (($# < 2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
      case "$1" in
        --dataset-dir) SOURCE_DIR="$2";;
        --outdir) FINAL_DIR="$2";;
        --figure-outdir) FIGURE_DIR="$2";;
        --n-jobs) JOBS="$2";;
        --n-estimators) TREES="$2";;
        --train-end) TRAIN_END="$2";;
        --validation-end) VALIDATION_END="$2";;
      esac
      shift 2;;
    --help|-h)
      printf 'Usage: bash scripts/run_repaired_project.sh [--dataset-dir SOURCE] [--outdir FINAL] [--figure-outdir FIGURES] [--n-jobs N] [--n-estimators N] [--train-end DATE] [--validation-end DATE]\n'
      exit 0;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2;;
  esac
done
if [[ ! -x "$SEP_PYTHON" ]]; then printf 'Run bash scripts/setup_env.sh first.\n' >&2; exit 1; fi
mkdir -p logs
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
"$SEP_PYTHON" -u -m sep_dataset.cli repair --dataset-dir "$SOURCE_DIR" --outdir "$FINAL_DIR" 2>&1 | tee "logs/${RUN_STAMP}_repair.log"
"$SEP_PYTHON" -u -m sep_dataset.cli validate --dataset-dir "$FINAL_DIR" 2>&1 | tee "logs/${RUN_STAMP}_validate.log"
"$SEP_PYTHON" -u -m sep_dataset.cli plot --dataset-dir "$FINAL_DIR" --outdir "$FIGURE_DIR" --with-models \
  --n-jobs "$JOBS" --n-estimators "$TREES" --train-end "$TRAIN_END" --validation-end "$VALIDATION_END" 2>&1 | tee "logs/${RUN_STAMP}_figures_models.log"
printf '\nDatasets: %s\nFigures: %s\nModel metrics and compact plots: %s/models\n' "$FINAL_DIR" "$FIGURE_DIR" "$FIGURE_DIR"

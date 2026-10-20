#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SEP_VENV_PATH="${SEP_VENV_DIR:-$PROJECT_DIR/.venv}"
END_DATE="2025-12-31"
DATASET_DIR="data/processed/compatible_1998_2025"
FIGURE_DIR="reports/figures_1998_2025"
BASELINE_DIR=""
INCLUDE_TSNE=0
usage() {
  printf 'Usage: bash scripts/run_project.sh [--baseline-dir PATH] [--end-date YYYY-MM-DD] [--outdir PATH] [--figure-outdir PATH] [--tsne]\n'
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-dir|--end-date|--outdir|--figure-outdir)
      if [[ $# -lt 2 || -z "$2" ]]; then usage >&2; exit 2; fi
      case "$1" in
        --baseline-dir) BASELINE_DIR="$2" ;;
        --end-date) END_DATE="$2" ;;
        --outdir) DATASET_DIR="$2" ;;
        --figure-outdir) FIGURE_DIR="$2" ;;
      esac
      shift 2 ;;
    --tsne) INCLUDE_TSNE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! -x "$SEP_VENV_PATH/bin/python" ]]; then
  printf 'Missing project environment. First run: bash scripts/setup_env.sh\n' >&2
  exit 1
fi
cd "$PROJECT_DIR"
mkdir -p logs
RUN_STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
BUILD_ARGS=(-u -m sep_dataset.cli build --end-date "$END_DATE" --outdir "$DATASET_DIR")
if [[ -n "$BASELINE_DIR" ]]; then BUILD_ARGS+=(--baseline-dir "$BASELINE_DIR"); fi
PLOT_ARGS=(-u -m sep_dataset.cli plot --dataset-dir "$DATASET_DIR" --outdir "$FIGURE_DIR")
if [[ "$INCLUDE_TSNE" == 1 ]]; then PLOT_ARGS+=(--tsne); fi
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
"$SEP_VENV_PATH/bin/python" "${BUILD_ARGS[@]}" \
  2>&1 | tee "logs/${RUN_STAMP}_build.log"
"$SEP_VENV_PATH/bin/python" -m sep_dataset.cli validate \
  --dataset-dir "$DATASET_DIR" \
  2>&1 | tee "logs/${RUN_STAMP}_validate.log"
"$SEP_VENV_PATH/bin/python" "${PLOT_ARGS[@]}" \
  2>&1 | tee "logs/${RUN_STAMP}_figures.log"
printf '\n[COMPLETE] Datasets: %s\nFigures: %s\nLogs: logs/%s_*\n' "$DATASET_DIR" "$FIGURE_DIR" "$RUN_STAMP"

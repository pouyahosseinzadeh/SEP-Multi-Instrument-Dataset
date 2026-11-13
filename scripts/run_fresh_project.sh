#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${SEP_PYTHON:-${SEP_VENV_DIR:-$project_dir/.venv}/bin/python}"
work_dir="data/fresh_1998_2025"
outdir="data/processed/recollected_ten_1998_2025"
mask_path=""
baseline_dir=""

while (($#)); do
  case "$1" in
    --work-dir|--outdir|--baseline-dir|--historical-masks)
      if (($# < 2)) || [[ -z "$2" ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --work-dir) work_dir="$2";;
        --outdir) outdir="$2";;
        --baseline-dir) baseline_dir="$2";;
        --historical-masks) mask_path="$2";;
      esac
      shift 2;;
    --help|-h)
      printf 'Usage: bash scripts/run_fresh_project.sh [--work-dir PATH] [--outdir PATH] [--baseline-dir PATH] [--historical-masks PATH]\n'
      exit 0;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2;;
  esac
done

if [[ ! -x "$python_bin" ]]; then
  printf 'Run bash scripts/setup_env.sh first.\n' >&2
  exit 1
fi

mkdir -p logs
log="logs/fresh_$(date -u +%Y%m%dT%H%M%SZ).log"
collect_args=(--work-dir "$work_dir")
repair_args=(--dataset-dir "$work_dir/native" --outdir "$outdir")
if [[ -n "$baseline_dir" ]]; then collect_args+=(--baseline-dir "$baseline_dir"); fi
if [[ -n "$mask_path" ]]; then
  collect_args+=(--historical-masks "$mask_path")
  repair_args+=(--historical-masks "$mask_path")
fi
{
  "$python_bin" -u -m sep_dataset.cli collect "${collect_args[@]}"
  "$python_bin" -u -m sep_dataset.cli repair "${repair_args[@]}"
  "$python_bin" -u -m sep_dataset.cli validate --dataset-dir "$outdir"
  printf '\n[COMPLETE] Ten validated feature files: %s\n' "$outdir"
} 2>&1 | tee "$log"

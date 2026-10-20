#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEP_VENV_PATH="${SEP_VENV_DIR:-$PROJECT_DIR/.venv}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\nSet PYTHON_BIN to Python 3.8 or newer.\n' "$PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; sys.exit("Python 3.8 or newer is required") if sys.version_info < (3, 8) else None'
bootstrap_virtualenv() {
  printf 'The selected Python lacks a working venv/ensurepip. Trying a user-local virtualenv bootstrap.\n' >&2
  if ! "$PYTHON_BIN" -m virtualenv --version >/dev/null 2>&1; then
    if ! "$PYTHON_BIN" -m pip install --user virtualenv; then
      printf 'Unable to bootstrap virtualenv. Set PYTHON_BIN to a Python 3.8+ installation with working pip and rerun this script. No sudo is required.\n' >&2
      exit 1
    fi
  fi
  "$PYTHON_BIN" -m virtualenv "$SEP_VENV_PATH"
}
if [[ ! -x "$SEP_VENV_PATH/bin/python" ]]; then
  if ! "$PYTHON_BIN" -m venv "$SEP_VENV_PATH"; then bootstrap_virtualenv; fi
fi
"$SEP_VENV_PATH/bin/python" -c 'import sys; sys.exit("Existing virtual environment needs Python 3.8 or newer") if sys.version_info < (3, 8) else None'
if ! "$SEP_VENV_PATH/bin/python" -m pip --version >/dev/null 2>&1; then
  if ! "$SEP_VENV_PATH/bin/python" -m ensurepip --upgrade; then bootstrap_virtualenv; fi
fi
"$SEP_VENV_PATH/bin/python" -m pip install --upgrade pip
"$SEP_VENV_PATH/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
printf '\nEnvironment ready: %s\nRun: bash scripts/run_project.sh\n' "$SEP_VENV_PATH"

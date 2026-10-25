#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$version" == "3.11" || "$version" == "3.12" ]]; then PYTHON_BIN=$(command -v "$candidate"); break; fi
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.11 or 3.12 is required. Install Python once, then reopen this folder in VS Code."
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating a private Python environment..."
  "$PYTHON_BIN" -m venv .venv
fi
if [[ ! -f .venv/.smartcrop-dependencies-installed ]]; then
  echo "Installing application libraries (first launch only; internet required)..."
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt || {
    echo "Dependency installation failed. Check your internet connection and Python version, then run the task again."
    exit 1
  }
  touch .venv/.smartcrop-dependencies-installed
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env. Gemini is optional; without a working key/quota the app uses built-in guidance."
fi
set -a
source .env
set +a
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"

echo "Starting Smart Crop Disease Detection..."
echo "Open http://127.0.0.1:5000 in your browser. Press Ctrl+C here to stop."
exec .venv/bin/python app.py

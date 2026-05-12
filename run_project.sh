set -euo pipefail

RESET_DB="${RESET_DB:-0}"
if [ "${1:-}" = "--reset-db" ]; then
  RESET_DB=1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required but was not found on PATH."
  exit 1
fi

VENV_DIR=".venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
REQUIREMENTS_STAMP="$VENV_DIR/.requirements_installed"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/oncolink-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo "  OncoLink — Precision Medicine Platform"
echo "============================================"

if [ "$RESET_DB" = "1" ]; then
  echo ""
  echo "[reset] Clearing local login/patient database..."
  rm -f outputs_metabric/oncolink.db
fi

echo ""
echo "[1/4] Installing dependencies..."
if [ ! -x "$PYTHON_BIN" ]; then
  echo "      Creating virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if [ ! -f "$REQUIREMENTS_STAMP" ] || [ requirements.txt -nt "$REQUIREMENTS_STAMP" ]; then
  "$PYTHON_BIN" -m pip install --upgrade pip --quiet
  "$PIP_BIN" install -r requirements.txt --quiet
  touch "$REQUIREMENTS_STAMP"
else
  echo "      Dependencies already installed in $VENV_DIR"
fi

echo ""
echo "[2/4] Processing METABRIC dataset..."
"$PYTHON_BIN" processing.py

echo ""
echo "[3/4] Training models (XGBoost + RF + LR + incremental)..."
"$PYTHON_BIN" model.py

echo ""
echo "[4/4] Launching Streamlit app..."
echo "      Open http://localhost:8501 in your browser"
echo ""
"$PYTHON_BIN" -m streamlit run app.py

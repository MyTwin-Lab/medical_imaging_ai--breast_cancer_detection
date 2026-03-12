#!/usr/bin/env bash
# =============================================================================
# setup.sh — Create the conda environment and download the INbreast dataset.
#
# Usage:
#   bash setup.sh [OPTIONS]
#
# Options:
#   --skip-env        Skip conda env creation (env already exists)
#   --skip-download   Skip dataset download (data already present)
# =============================================================================

set -eo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
die()  { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

SKIP_ENV=0
SKIP_DOWNLOAD=0
ENV_NAME="mamography"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for arg in "$@"; do
  case $arg in
    --skip-env)      SKIP_ENV=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *) die "Unknown option: $arg" ;;
  esac
done

echo -e "\n${BOLD}========================================${RESET}"
echo -e "${BOLD}  Breast Cancer Detection — Setup       ${RESET}"
echo -e "${BOLD}========================================${RESET}\n"

cd "$SCRIPT_DIR"

# =============================================================================
# 1. Conda environment
# =============================================================================
log "Step 1/2 — Conda environment"

CONDA_SH=""
for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
  if [[ -f "$candidate" ]]; then
    CONDA_SH="$candidate"
    break
  fi
done
[[ -z "$CONDA_SH" ]] && die "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"

source "$CONDA_SH"

if [[ $SKIP_ENV -eq 1 ]]; then
  warn "Skipping env creation (--skip-env)"
else
  if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Env '${ENV_NAME}' already exists — removing it first ..."
    conda env remove -n "$ENV_NAME" -y
  fi
  log "Creating conda env '${ENV_NAME}' from environment.yml ..."
  conda env create -f environment.yml
fi

ok "Env '${ENV_NAME}' ready"

# GPU check
GPU=$(conda run -n "$ENV_NAME" python -c \
  "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" 2>/dev/null || echo "NONE")
if [[ "$GPU" == "NONE" ]]; then
  warn "No CUDA GPU detected — training will be very slow on CPU"
else
  ok "GPU: $GPU"
fi

# =============================================================================
# 2. Dataset download
# =============================================================================
log "Step 2/2 — Dataset"

if [[ $SKIP_DOWNLOAD -eq 1 ]]; then
  warn "Skipping download (--skip-download)"
else
  DICOM_DIR="$SCRIPT_DIR/INbreast/AllDICOMs"
  if [[ -d "$DICOM_DIR" ]] && [[ -n "$(ls -A "$DICOM_DIR" 2>/dev/null)" ]]; then
    warn "INbreast/AllDICOMs already populated — skipping download"
  else
    [[ ! -f "$SCRIPT_DIR/.env" ]] && die ".env not found. Add KAGGLE_USERNAME and KAGGLE_KEY to .env"
    log "Downloading INbreast dataset from Kaggle ..."
    conda run -n "$ENV_NAME" python download_inbreast.py
  fi
fi

ok "Dataset ready"

# =============================================================================
# Done
# =============================================================================
echo -e "\n${BOLD}${GREEN}Setup complete!${RESET}"
echo -e "  Activate the environment:  ${BOLD}conda activate ${ENV_NAME}${RESET}"
echo -e "  Then see README.md for training and inference commands.\n"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STATE_DATASET="${STATE_DATASET:-TX}"
FILE_NAME="${FILE_NAME:-house_id_da09897f6b67c4511ee33c658ddbdfe3afd082e3.csv}"
RC_MODEL="${RC_MODEL:-R1C1}"
DATASET_RAW_DIR="${DATASET_RAW_DIR:-$PROJECT_ROOT/data/raw/ecobee/house_data_by_state}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
DEVICE="${DEVICE:-cuda:0}"

cd "$PROJECT_ROOT"

python main.py \
  --state_dataset "$STATE_DATASET" \
  --file_name "$FILE_NAME" \
  --rc_model "$RC_MODEL" \
  --dataset_raw_dir "$DATASET_RAW_DIR" \
  --pretrain_epochs "$PRETRAIN_EPOCHS" \
  --max_epochs "$MAX_EPOCHS" \
  --device "$DEVICE"

#!/bin/bash
set -euo pipefail

echo "==> Starting base image services (Jupyter/SSH) in background..."
/start.sh &

echo "==> Wait for services to start..."
sleep 5

# --- Repository setup ---
REPO_URL="${REPO_URL:?REPO_URL env var is required}"
REPO_DIR="${REPO_DIR:-/app/repo}"
REPO_BRANCH="${REPO_BRANCH:?REPO_BRANCH env var is required}"
MODEL="${MODEL:?MODEL env var is required}"

# --- Training setup ---
DATASET_PATH="${DATASET_PATH:-/workspace/dataset}"
SAVE_PATH="${SAVE_PATH:-/workspace/results}"
WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY env var is required}"
NUM_BRAINS="${NUM_BRAINS:?NUM_BRAINS env var is required}"
BATCH_SIZE="${BATCH_SIZE:?BATCH_SIZE env var is required}"
LR="${LR:?LR env var is required}"
NUM_EPOCHS="${NUM_EPOCHS:?NUM_EPOCHS env var is required}"

# --- Export so training script inherits them ---
export DATASET_PATH
export SAVE_PATH
export WANDB_API_KEY
export NUM_BRAINS
export BATCH_SIZE
export LR
export NUM_EPOCHS

echo "==> Cloning repo: $REPO_URL (branch: $REPO_BRANCH)"
git clone --branch "$REPO_BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

# --- Virtual Enviroment setup ---
echo "==> Syncing dependencies..."
uv sync --locked || uv sync

# --- Log into W&B
#echo "==> Logging into Weights & Biases..."
#uv run wandb login "$WANDB_API_KEY"

# --- Create path if it doesn't exist ---
mkdir -p "$SAVE_PATH"

echo "==> Running: $MODEL"
echo "    DATASET_PATH=$DATASET_PATH"
echo "    SAVE_PATH=$SAVE_PATH"

exec uv run python "$REPO_DIR/models/$MODEL/src/train.py" || {
    echo "==> Training failed! Keeping container alive..."
    tail -f /dev/null
}

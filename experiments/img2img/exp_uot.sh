#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─────────────────────────  CONFIG  ─────────────────────────
AVAILABLE_GPUS=(6)                 # physical GPU IDs
CONCURRENCY_PER_GPU=1              # → TOTAL_SLOTS = 4, jobs run round-robin
NUM_GPUS=${#AVAILABLE_GPUS[@]}
TOTAL_SLOTS=$(( NUM_GPUS * CONCURRENCY_PER_GPU ))

CONDA_ENV_NAME='uot-fm'               # conda env with your code
ENTRY_SCRIPT='uot_alae.py'         # main script
RUN_DIR='runs_uot'                 # log directory
mkdir -p "$RUN_DIR"

# ─────────────────────────  PYTHON  ─────────────────────────
if ! command -v conda &>/dev/null; then
  echo 'Error: conda not in PATH' >&2 ; exit 1
fi
CONDA_BASE=$(conda info --base 2>/dev/null)
PYTHON_EXE="$CONDA_BASE/envs/$CONDA_ENV_NAME/bin/python"
[[ -x $PYTHON_EXE ]] || {
  PYTHON_EXE=$(conda run -n "$CONDA_ENV_NAME" which python) || {
    echo "Cannot find python in env '$CONDA_ENV_NAME'." >&2 ; exit 1; }
}
echo "Using Python : $PYTHON_EXE"
echo "Using GPUs   : ${AVAILABLE_GPUS[*]}  (total $NUM_GPUS)"

command -v parallel >/dev/null   || { echo "GNU Parallel missing." >&2; exit 1; }
[[ -f $ENTRY_SCRIPT ]]           || { echo "$ENTRY_SCRIPT not found in $(pwd)." >&2; exit 1; }

# ─────────────────────  BUILD ARGUMENT LIST  ─────────────────
ARGS_FILE=$(mktemp)
trap 'rm -f "$ARGS_FILE"' EXIT

declare -a EXPERIMENTS=(
  # ADULT TO YOUNG
  '--method uot-fm --input ADULT --target YOUNG --num_epoch 1000 --ufm_eps 0.1 --wandb'
  '--method uot-fm --input ADULT --target YOUNG --num_epoch 1000 --ufm_eps 0.05 --wandb'
  '--method uot-fm --input ADULT --target YOUNG --num_epoch 1000 --ufm_eps 0.005 --wandb'
  '--method uot-fm --input ADULT --target YOUNG --num_epoch 1000 --ufm_eps 0.001 --wandb'
  '--method uot-fm --input ADULT --target YOUNG --num_epoch 1000 --ufm_eps 0.0005 --wandb'
  '--method uot-fm --input ADULT --target YOUNG --num_epoch 1000 --ufm_eps 0.0001 --wandb'
  
  # MAN TO WOMAN
  '--method uot-fm --input MAN --target WOMAN --num_epoch 1000 --ufm_eps 0.1 --wandb'
  '--method uot-fm --input MAN --target WOMAN --num_epoch 1000 --ufm_eps 0.05 --wandb'
  '--method uot-fm --input MAN --target WOMAN --num_epoch 1000 --ufm_eps 0.005 --wandb'
  '--method uot-fm --input MAN --target WOMAN --num_epoch 1000 --ufm_eps 0.001 --wandb'
  '--method uot-fm --input MAN --target WOMAN --num_epoch 1000 --ufm_eps 0.0005 --wandb'
  '--method uot-fm --input MAN --target WOMAN --num_epoch 1000 --ufm_eps 0.0001 --wandb'
  
  # WOMAN TO MAN
  '--method uot-fm --input WOMAN --target MAN --num_epoch 1000 --ufm_eps 0.1 --wandb'
  '--method uot-fm --input WOMAN --target MAN --num_epoch 1000 --ufm_eps 0.05 --wandb'
  '--method uot-fm --input WOMAN --target MAN --num_epoch 1000 --ufm_eps 0.005 --wandb'
  '--method uot-fm --input WOMAN --target MAN --num_epoch 1000 --ufm_eps 0.001 --wandb'
  '--method uot-fm --input WOMAN --target MAN --num_epoch 1000 --ufm_eps 0.0005 --wandb'
  '--method uot-fm --input WOMAN --target MAN --num_epoch 1000 --ufm_eps 0.0001 --wandb'
)

# Write each experiment to ARGS_FILE
for args in "${EXPERIMENTS[@]}"; do
  echo "$args" >>"$ARGS_FILE"
done

echo -e "\n=== Argument lines (cat -A) ==="
cat -A "$ARGS_FILE"
echo '================================'

export ENTRY_SCRIPT PYTHON_EXE NUM_GPUS RUN_DIR AVAILABLE_GPUS_STR="${AVAILABLE_GPUS[*]}"

# ─────────────────────  GNU PARALLEL LAUNCH  ─────────────────
parallel -j "$TOTAL_SLOTS" --line-buffer --halt soon,fail=1 \
  --joblog "$RUN_DIR/joblog.tsv" \
  '
    slot={%}      # 1 … TOTAL_SLOTS   –> "worker slot"
    job={#}       # 1 … #lines        –> unique job counter

    # Map slot → real GPU:
    GPUS=($AVAILABLE_GPUS_STR)
    export CUDA_VISIBLE_DEVICES="${GPUS[(( (slot-1) % NUM_GPUS ))]}"

    out="$RUN_DIR/job_${job}_stdout.log"
    err="$RUN_DIR/job_${job}_stderr.log"

    ARGS=$(echo {} | tr -d "'\''")   # strip GNU Parallel quotes
    echo "[Job $job / Slot $slot | GPU $CUDA_VISIBLE_DEVICES] $PYTHON_EXE $ENTRY_SCRIPT $ARGS"
    eval "$PYTHON_EXE $ENTRY_SCRIPT $ARGS" >"$out" 2>"$err"
  ' :::: "$ARGS_FILE"

echo "✓ All jobs finished. Logs in '$RUN_DIR' and $RUN_DIR/joblog.tsv" 
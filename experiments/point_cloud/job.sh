#!/usr/bin/env bash
rm -rf results
rm -rf runs_uot
set -euo pipefail
IFS=$'\n\t'

# ─────────────────────────  CONFIG  ─────────────────────────
AVAILABLE_GPUS=(1 2 3)                 # physical GPU IDs
CONCURRENCY_PER_GPU=1              # → TOTAL_SLOTS = 4, jobs run round-robin
NUM_GPUS=${#AVAILABLE_GPUS[@]}
TOTAL_SLOTS=$(( NUM_GPUS * CONCURRENCY_PER_GPU ))

CONDA_ENV_NAME='partial'              # conda env with your code
ENTRY_SCRIPT='main.py'            # ← changed from uot_alae.py
RUN_DIR='runs_uot'                      # log directory
EPS=500
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
  # Partial-TWD
  '--method twd --twd_unbalanced --max_mass_generated 0.9'
  '--method twd --twd_unbalanced --max_mass_generated 0.95'
  '--method twd --twd_unbalanced --max_mass_generated 0.96'
  '--method twd --twd_unbalanced --max_mass_generated 0.99'
  '--method twd --twd_unbalanced --max_mass_generated 1.0'

  # SW
  # '--method sw'

  # PAWL  (k = 256 … 10)
  # best
  # '--method pawl --pawl_k 256'
  # '--method pawl --pawl_k 200'
  # '--method pawl --pawl_k 100'
  # '--method pawl --pawl_k 75'
  # '--method pawl --pawl_k 50'
  # '--method pawl --pawl_k 25'
  # '--method pawl --pawl_k 10 --lr 0.001'
  # '--method pawl --pawl_k 10 --lr 0.01'
  # '--method pawl --pawl_k 10 --lr 0.1'

  # USOT (ρ₁, ρ₂)
  # '--method usot --rho1 0.01 --rho2 0.01'
  # '--method usot --rho1 1    --rho2 1'
  # '--method usot --rho1 100  --rho2 100'
  # '--method usot --rho1 200  --rho2 200'
  # '--method usot --rho1 300  --rho2 300'
  # '--method usot --rho1 400  --rho2 400'
  # '--method usot --rho1 500  --rho2 500'
  # '--method usot --rho1 1000  --rho2 1000'
  # '--method usot --rho1 10000  --rho2 10000'
  # '--method usot --rho1 1000  --rho2 1'
  # '--method usot --rho1 1  --rho2 1000'
  
  # SUOT (ρ₁, ρ₂)
  # '--method suot --rho1 0.01 --rho2 0.01'
  # '--method suot --rho1 1    --rho2 1'
  # '--method suot --rho1 100  --rho2 100'
  # '--method suot --rho1 200  --rho2 200'
  # '--method suot --rho1 300  --rho2 300'
  # '--method suot --rho1 400  --rho2 400'
  # '--method suot --rho1 500  --rho2 500'
  # '--method suot --rho1 1000  --rho2 1000'
  # '--method suot --rho1 10000  --rho2 10000'
  # '--method suot --rho1 1000  --rho2 1'
  # '--method suot --rho1 1  --rho2 1000'

  # SOPT
  # '--method sopt --sopt_reg 0.01'
  # '--method sopt --sopt_reg 0.05'
  # '--method sopt --sopt_reg 0.1'
  # '--method sopt --sopt_reg 0.5'
  # '--method sopt --sopt_reg 1'
  # '--method sopt --sopt_reg 10'
  # '--method sopt --sopt_reg 100'

  # SPOT
  # '--method spot --spot_k 256'
  # '--method spot --spot_k 150'
  # '--method spot --spot_k 160'
  # '--method spot --spot_k 170'
  # '--method spot --spot_k 180'
  # '--method spot --spot_k 190'
  # '--method spot --spot_k 210'
  # '--method spot --spot_k 220'
  # '--method spot --spot_k 230'
  # '--method spot --spot_k 200'
  # '--method spot --spot_k 100'
  # '--method spot --spot_k 75'
  # '--method spot --spot_k 50'
  # '--method spot --spot_k 25'
  # '--method spot --spot_k 10 --lr 0.001'
  # '--method spot --spot_k 10 --lr 0.01'
  # '--method spot --spot_k 10 --lr 0.1'

  # Faster-UOT (POT)
  # '--method pot --pot_reg 0.2 --pot_reg_m_kl 0.2'
  # '--method pot --pot_reg 0.3 --pot_reg_m_kl 0.3'
  # '--method pot --pot_reg 0.4 --pot_reg_m_kl 0.4'
#   '--method pot --pot_reg 0.5 --pot_reg_m_kl 0.5'
#   '--method pot --pot_reg 0.7 --pot_reg_m_kl 0.7'
#   '--method pot --pot_reg 0.9 --pot_reg_m_kl 0.9'
#   '--method pot --pot_reg 0.5 --pot_reg_m_kl 0.1'
#   '--method pot --pot_reg 0.1 --pot_reg_m_kl 0.5'
#   '--method pot --pot_reg 0.3 --pot_reg_m_kl 0.5'
  # '--method pot --pot_reg 0.5 --pot_reg_m_kl 0.3 --lr 0.001'
  # '--method pot --pot_reg 0.5 --pot_reg_m_kl 0.3 --lr 0.01'
  # '--method pot --pot_reg 0.5 --pot_reg_m_kl 0.3 --lr 0.1'
  # '--method pot --pot_reg 0.5 --pot_reg_m_kl 0.3 --lr 1.0'
  # '--method pot --pot_reg 0.5 --pot_reg_m_kl 0.3 --lr 10.0'
#   '--method pot --pot_reg 0.9 --pot_reg_m_kl 0.1'
#   '--method pot --pot_reg 0.1 --pot_reg_m_kl 0.9'
)

# Write each experiment to ARGS_FILE
for args in "${EXPERIMENTS[@]}"; do
  echo "$args --num_epoch $EPS" >>"$ARGS_FILE"
done

echo -e "\n=== Argument lines (cat -A) ==="
cat -A "$ARGS_FILE"
echo '================================'

export ENTRY_SCRIPT PYTHON_EXE NUM_GPUS RUN_DIR AVAILABLE_GPUS_STR="${AVAILABLE_GPUS[*]}"

# ─────────────────────  GNU PARALLEL LAUNCH  ─────────────────
parallel -j "$TOTAL_SLOTS" --line-buffer --halt never \
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

# plot results
python visus_shape.py --method twd --plot_style 1
python visus_shape.py --method twd --plot_style 2
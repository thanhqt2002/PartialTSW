#!/usr/bin/env bash
set -euo pipefail      # abort on error, unset vars, or pipe fails
IFS=$'\n\t'

# ─────────────────────────  CONFIG  ─────────────────────────
export WANDB_API_KEY=$WANDB_API_KEY

AVAILABLE_GPUS=(6)                     # physical GPU IDs
CONCURRENCY_PER_GPU=1                   # ← run several jobs at a time on each GPU
NUM_GPUS=${#AVAILABLE_GPUS[@]}
TOTAL_SLOTS=$(( NUM_GPUS * CONCURRENCY_PER_GPU ))
CONDA_ENV_NAME='partial'               # conda env with your code
RUN_DIR='runs'                       # where logs are saved
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
[[ -f uot_alae.py ]]             || { echo "uot_alae.py not in $(pwd)." >&2; exit 1; }

# ─────────────────────  BUILD ARGUMENT LIST  ─────────────────
ARGS_FILE=$(mktemp)
trap 'rm -f "$ARGS_FILE"' EXIT


cat >"$ARGS_FILE"<<'EOF'
--method ulight --ulight_tau 50 --input YOUNG --target ADULT --num_epoch 5000 --wandb
--method twd --input YOUNG --target ADULT --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1  --wandb
--method ulight --ulight_tau 50 --input ADULT --target YOUNG --num_epoch 5000 --wandb
--method ulight --ulight_tau 100 --input ADULT --target YOUNG --num_epoch 5000 --wandb
--method ulight --ulight_tau 250 --input ADULT --target YOUNG --num_epoch 5000 --wandb
--method ulight --ulight_tau 1000 --input ADULT --target YOUNG --num_epoch 5000 --wandb
--method ulight --ulight_tau 10000 --input ADULT --target YOUNG --num_epoch 5000 --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1  --wandb
--method sw --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.05 --twd_max_total_mass_Y 1  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.1 --twd_max_total_mass_Y 1  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.2 --twd_max_total_mass_Y 1  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.3 --twd_max_total_mass_Y 1  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.5 --twd_max_total_mass_Y 1  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.9 --twd_max_total_mass_Y 1  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.1  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.2  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.5  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 2  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 2.5  --wandb
--method twd --input ADULT --target YOUNG --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 3  --wandb
--method ulight --ulight_tau 50 --input MAN --target WOMAN --num_epoch 5000 --wandb
--method ulight --ulight_tau 100 --input MAN --target WOMAN --num_epoch 5000 --wandb
--method ulight --ulight_tau 250 --input MAN --target WOMAN --num_epoch 5000 --wandb
--method ulight --ulight_tau 1000 --input MAN --target WOMAN --num_epoch 5000 --wandb
--method ulight --ulight_tau 10000 --input MAN --target WOMAN --num_epoch 5000 --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1  --wandb
--method sw --input MAN --target WOMAN --num_epoch 1000 --L 1000 --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.05 --twd_max_total_mass_Y 1  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.1 --twd_max_total_mass_Y 1  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.2 --twd_max_total_mass_Y 1  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.3 --twd_max_total_mass_Y 1  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.5 --twd_max_total_mass_Y 1  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.9 --twd_max_total_mass_Y 1  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.1  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.2  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.5  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 2  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 2.5  --wandb
--method twd --input MAN --target WOMAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 3  --wandb
--method ulight --ulight_tau 50 --input WOMAN --target MAN --num_epoch 5000 --wandb
--method ulight --ulight_tau 100 --input WOMAN --target MAN --num_epoch 5000 --wandb
--method ulight --ulight_tau 250 --input WOMAN --target MAN --num_epoch 5000 --wandb
--method ulight --ulight_tau 1000 --input WOMAN --target MAN --num_epoch 5000 --wandb
--method ulight --ulight_tau 10000 --input WOMAN --target MAN --num_epoch 5000 --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1  --wandb
--method sw --input WOMAN --target MAN --num_epoch 1000 --L 1000 --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.05 --twd_max_total_mass_Y 1  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.1 --twd_max_total_mass_Y 1  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.2 --twd_max_total_mass_Y 1  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.3 --twd_max_total_mass_Y 1  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.5 --twd_max_total_mass_Y 1  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 0.9 --twd_max_total_mass_Y 1  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.1  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.2  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 1.5  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 2  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 2.5  --wandb
--method twd --input WOMAN --target MAN --num_epoch 1000 --L 1000 --twd_nlines 4 --twd_unbalanced_scheduler constant          --twd_min_total_mass_Y 1 --twd_max_total_mass_Y 3  --wandb
EOF



declare -a EXPERIMENTS=(
  # Faster-UOT (POT)
  # '--method faster-uot --faster_uot_reg 0.001 --faster_uot_reg_m_kl 0.001 --wandb'
  # '--method faster-uot --faster_uot_reg 0.005 --faster_uot_reg_m_kl 0.005 --wandb'
  # '--method faster-uot --faster_uot_reg 0.01 --faster_uot_reg_m_kl 0.01 --wandb'
  # '--method faster-uot --faster_uot_reg 0.05 --faster_uot_reg_m_kl 0.05 --wandb'
  # '--method faster-uot --faster_uot_reg 0.1 --faster_uot_reg_m_kl 0.1 --wandb'
  # '--method faster-uot --faster_uot_reg 0.3 --faster_uot_reg_m_kl 0.3 --wandb'
  # '--method faster-uot --faster_uot_reg 0.4 --faster_uot_reg_m_kl 0.4 --wandb'
  
  # # PAWL (k = 256 … 10)
  # '--method pawl --pawl_k 256 --wandb'
  # '--method pawl --pawl_k 200 --wandb'
  # '--method pawl --pawl_k 100 --wandb'
  # '--method pawl --pawl_k 75 --wandb'
  # '--method pawl --pawl_k 50 --wandb'
  # '--method pawl --pawl_k 25 --wandb'
  # '--method pawl --pawl_k 10 --wandb'

  # # USOT (ρ₁, ρ₂)
  # '--method usot --rho1 0.01 --rho2 0.01 --wandb'
  # '--method usot --rho1 1    --rho2 1    --wandb'
  # '--method usot --rho1 100  --rho2 100  --wandb'
  # '--method usot --rho1 200  --rho2 200  --wandb'
  # '--method usot --rho1 300  --rho2 300  --wandb'
  # '--method usot --rho1 400  --rho2 400  --wandb'
  # '--method usot --rho1 500  --rho2 500  --wandb'
  # '--method usot --rho1 1000  --rho2 1000  --wandb'
  
  # # SUOT (ρ₁, ρ₂)
  # '--method suot --rho1 0.01 --rho2 0.01 --wandb'
  # '--method suot --rho1 1    --rho2 1    --wandb'
  # '--method suot --rho1 100  --rho2 100  --wandb'
  # '--method suot --rho1 200  --rho2 200  --wandb'
  # '--method suot --rho1 300  --rho2 300  --wandb'
  # '--method suot --rho1 400  --rho2 400  --wandb'
  # '--method suot --rho1 500  --rho2 500  --wandb'
  # '--method suot --rho1 1000  --rho2 1000  --wandb'

  # # SOPT
  # '--method sopt --sopt_reg 0.01 --wandb'
  # '--method sopt --sopt_reg 0.05 --wandb'
  # '--method sopt --sopt_reg 0.1  --wandb'
  # '--method sopt --sopt_reg 0.5  --wandb'
  # '--method sopt --sopt_reg 1    --wandb'
  # '--method sopt --sopt_reg 10   --wandb'
  # '--method sopt --sopt_reg 100  --wandb'

  # # SPOT
  # '--method spot --spot_k 256 --wandb'
  # '--method spot --spot_k 200 --wandb'
  # '--method spot --spot_k 150 --wandb'
  # '--method spot --spot_k 100 --wandb'
  # '--method spot --spot_k 75  --wandb'
  # '--method spot --spot_k 50  --wandb'
  # '--method spot --spot_k 25  --wandb'
  # '--method spot --spot_k 10  --wandb'
)

# Write each experiment to ARGS_FILE
for args in "${EXPERIMENTS[@]}"; do
  echo "$args" >>"$ARGS_FILE"
done

echo -e "\n=== Argument lines (cat -A) ==="
cat -A "$ARGS_FILE"
echo '================================'

export PYTHON_EXE NUM_GPUS RUN_DIR AVAILABLE_GPUS_STR="${AVAILABLE_GPUS[*]}"

# ─────────────────────  GNU PARALLEL LAUNCH  ─────────────────
parallel -j "$TOTAL_SLOTS" --line-buffer --halt soon,fail=1 \
  --joblog "$RUN_DIR/joblog.tsv" \
  '
    slot={%}           # 1 … TOTAL_SLOTS   –> "worker slot"
    job={#}            # 1 … #lines        –> unique job counter

    # map slot to a real GPU:
    GPUS=($AVAILABLE_GPUS_STR)
    export CUDA_VISIBLE_DEVICES="${GPUS[(( (slot-1) % NUM_GPUS ))]}"

    out="$RUN_DIR/job_${job}_stdout.log"
    err="$RUN_DIR/job_${job}_stderr.log"

    ARGS=$(echo {} | tr -d "'\''")   # remove Parallel’s quotes
    echo "[Job $job / Slot $slot | GPU $CUDA_VISIBLE_DEVICES] $PYTHON_EXE uot_alae.py $ARGS"
    eval "$PYTHON_EXE uot_alae.py $ARGS" >"$out" 2>"$err"
  ' :::: "$ARGS_FILE"

echo "✓ All jobs finished. Logs in '$RUN_DIR' and $RUN_DIR/joblog.tsv"

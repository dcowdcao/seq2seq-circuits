#!/bin/bash
# Training job script for DSL algorithms from the models.txt run list.
#
# This script decides WHICH tier to train and WHETHER to promote. It does NOT carry
# hyperparameters: the recipe per tier lives in hparams.py and train.py reads it directly,
# so this path and the vmap path (train_stacked.py) cannot disagree.
#
# Usage:
#   ./run_training_job.sh                    # Train all algorithms in models.txt
#   ./run_training_job.sh --dry-run          # Show what would be trained without running
#   ./run_training_job.sh algo1 algo2        # Train specific algorithms only
#   ./run_training_job.sh --file mylist.txt  # Train algorithms from a custom file
#   ./run_training_job.sh --config wide algo1        # start at a specific tier
#   ./run_training_job.sh --upgrade-config xlarge …  # arch-aware escalation target
#   ./run_training_job.sh --no-upgrade       # Disable the escalation attempt
#   ./run_training_job.sh --seeds 3 algo1    # 3 seeds; best promoted, others banked

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"                      # train.py + its imports resolve relative to here
# MODELS_DIR is models/ (one level up); passing checkpoints promote to trained_models_pt/.
MODELS_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_FILE="$MODELS_DIR/models.txt"
JOBS_DIR="$SCRIPT_DIR/jobs"
CHECKPOINTS_DIR="$SCRIPT_DIR/job_checkpoints"
SEED_BANK="$CHECKPOINTS_DIR/seeds"    # extra seeds land here, never in the registry tree
TRAINED_MODELS_PT="$MODELS_DIR/trained_models_pt"
PY="${PYTHON:-$(cd "$SCRIPT_DIR/../../.." && pwd)/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

CONFIG_INITIAL="large"
CONFIG_UPGRADE="xlarge"   # escalate to deeper+wider; its recipe comes from hparams.py too
# GPU count is throughput only: train.py splits hparams' EFFECTIVE batch across the ranks.
if [ -z "$NPROC" ]; then
    NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NPROC" -eq 0 ]; then
        NPROC=1
    fi
fi
# Use MASTER_PORT from environment if set, otherwise random port
if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT=$((29500 + RANDOM % 1000))
fi
ACCURACY_THRESHOLD=0.95
UPGRADE_THRESHOLD=0.80  # If >= 80% on the initial tier, retry with CONFIG_UPGRADE
NO_UPGRADE=false
SEEDS=1

# Job timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# train_one <algo> <config> <seed> [logfile] — one train.py attempt; the recipe is
# resolved inside train.py from hparams.for_tier(<config>).
train_one() {
    local algo=$1 config=$2 seed=$3 log_file=$4
    local cmd=(torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT train.py
        --algorithm "$algo"
        --allow_repeats
        --random_sampling
        --seed "$seed"
        --target "$ACCURACY_THRESHOLD"
        --config "$config"
        --output_dir "$CHECKPOINTS_DIR")
    if [ "$DRY_RUN" = true ]; then
        echo "${cmd[@]}"
        return 0
    fi
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$log_file"
    return ${PIPESTATUS[0]}
}

# best_acc <logfile> — the trainer's own best eval seq-acc, or empty when it never printed one
best_acc() {
    grep -oP "Best eval accuracy: \K[0-9.]+" "$1" | tail -1
}

# gt <a> <b> — exit 0 iff a > b; ge <a> <b> — exit 0 iff a >= b (bash has no float compare)
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a > b)}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a >= b)}'; }

# bank <algo> <seed> — keep a non-promoted seed's checkpoint for the reproducibility read
bank() {
    mkdir -p "$SEED_BANK"
    cp "$CHECKPOINTS_DIR/checkpoint_${1}_best.pt" "$SEED_BANK/${1}.seed${2}.pt"
}

# Create directories
mkdir -p "$JOBS_DIR"
mkdir -p "$CHECKPOINTS_DIR"

SUCCESS_FILE="$JOBS_DIR/success_${TIMESTAMP}.txt"
FAILURE_FILE="$JOBS_DIR/failure_${TIMESTAMP}.txt"

# Initialize output files
echo "# Training job started at $(date)" > "$SUCCESS_FILE"
echo "# Algorithms with accuracy >= ${ACCURACY_THRESHOLD}" >> "$SUCCESS_FILE"
echo "" >> "$SUCCESS_FILE"

echo "# Training job started at $(date)" > "$FAILURE_FILE"
echo "# Algorithms with accuracy < ${ACCURACY_THRESHOLD}" >> "$FAILURE_FILE"
echo "" >> "$FAILURE_FILE"

# Parse arguments
DRY_RUN=false
SPECIFIC_ALGOS=()
CUSTOM_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --file|-f)
            CUSTOM_FILE="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG_INITIAL="$2"
            shift 2
            ;;
        --no-upgrade)
            NO_UPGRADE=true
            shift
            ;;
        --upgrade-config)
            CONFIG_UPGRADE="$2"   # arch-aware escalation target from run_model_pipeline
            shift 2
            ;;
        --seeds)
            SEEDS="$2"            # n=1 is the corpus default; >1 measures refit noise
            shift 2
            ;;
        *)
            SPECIFIC_ALGOS+=("$1")
            shift
            ;;
    esac
done

# Use custom file if specified
if [ -n "$CUSTOM_FILE" ]; then
    if [ -f "$CUSTOM_FILE" ]; then
        MODELS_FILE="$CUSTOM_FILE"
    elif [ -f "$MODELS_DIR/$CUSTOM_FILE" ]; then
        MODELS_FILE="$MODELS_DIR/$CUSTOM_FILE"
    elif [ -f "$SCRIPT_DIR/$CUSTOM_FILE" ]; then
        MODELS_FILE="$SCRIPT_DIR/$CUSTOM_FILE"
    else
        echo "Error: File not found: $CUSTOM_FILE"
        exit 1
    fi
fi

# Get algorithms to train
if [ ${#SPECIFIC_ALGOS[@]} -gt 0 ]; then
    ALGORITHMS=("${SPECIFIC_ALGOS[@]}")
else
    # Read from the models.txt run list, skip comments and empty lines
    ALGORITHMS=()
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        ALGORITHMS+=("$line")
    done < "$MODELS_FILE"
fi

echo "=========================================="
echo "Training Job: $TIMESTAMP"
echo "=========================================="
echo "Run list: $MODELS_FILE"
echo "Checkpoints dir: $CHECKPOINTS_DIR"
echo "Jobs dir: $JOBS_DIR"
echo "Initial config: $CONFIG_INITIAL"
echo "Upgrade config: $CONFIG_UPGRADE (if acc >= $UPGRADE_THRESHOLD)"
echo "Seeds per algorithm: $SEEDS"
echo "Success threshold: $ACCURACY_THRESHOLD"
echo "Algorithms to train: ${#ALGORITHMS[@]}"
echo ""
echo "--- resolved recipe (hparams.py; identical for train_stacked.py) ---"
"$PY" hparams.py --tier "$CONFIG_INITIAL"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - Commands that would be executed:"
    echo ""
    for algo in "${ALGORITHMS[@]}"; do
        echo "# First attempt with $CONFIG_INITIAL:"
        train_one "$algo" "$CONFIG_INITIAL" 0
        echo ""
        echo "# If acc in [$UPGRADE_THRESHOLD, $ACCURACY_THRESHOLD), retry with $CONFIG_UPGRADE"
        if [ "$NO_UPGRADE" = true ]; then
            echo "# (upgrade disabled by --no-upgrade)"
        fi
        echo "# Promotion (the ONE gate; refuses an unmeasured or sub-gate ckpt):"
        echo "$PY promote.py $CHECKPOINTS_DIR/checkpoint_${algo}_best.pt $TRAINED_MODELS_PT/${algo}.pt --gate $ACCURACY_THRESHOLD"
        echo ""
    done
    exit 0
fi

# Training loop
TOTAL=${#ALGORITHMS[@]}
CURRENT=0
SUCCESSES=0
FAILURES=0
UPGRADES=0

for algo in "${ALGORITHMS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Training: $algo"
    echo "=========================================="

    BEST_ACC=""
    FINAL_CONFIG=""
    BEST_SEED=""

    for (( SEED=0; SEED<SEEDS; SEED++ )); do
        echo ""
        echo "--- Attempt 1: config=$CONFIG_INITIAL seed=$SEED ---"
        TRAIN_LOG="$JOBS_DIR/train_${algo}_${CONFIG_INITIAL}_s${SEED}_${TIMESTAMP}.log"

        set +e
        train_one "$algo" "$CONFIG_INITIAL" "$SEED" "$TRAIN_LOG"
        TRAIN_EXIT_CODE=$?
        set -e

        if [ $TRAIN_EXIT_CODE -ne 0 ]; then
            echo "Training failed for $algo (seed $SEED)"
            echo "$algo  # training_failed_${CONFIG_INITIAL}_s${SEED}" >> "$FAILURE_FILE"
            continue
        fi

        SEED_ACC=$(best_acc "$TRAIN_LOG")
        SEED_CONFIG=$CONFIG_INITIAL
        if [ -z "$SEED_ACC" ]; then
            echo "Could not extract accuracy for $algo (seed $SEED)"
            echo "$algo  # accuracy_unknown_s${SEED}" >> "$FAILURE_FILE"
            continue
        fi
        echo ""
        echo "Best accuracy on $CONFIG_INITIAL (seed $SEED): $SEED_ACC"

        # Escalate to the arch-aware upgrade target when acc lands in [0.80, 0.95)
        if [ "$NO_UPGRADE" = false ] && ge "$SEED_ACC" "$UPGRADE_THRESHOLD" \
            && ! ge "$SEED_ACC" "$ACCURACY_THRESHOLD"; then
            echo ""
            echo "--- Attempt 2: Upgrading to config=$CONFIG_UPGRADE (seed $SEED) ---"
            UPGRADES=$((UPGRADES + 1))
            TRAIN_LOG_XLARGE="$JOBS_DIR/train_${algo}_${CONFIG_UPGRADE}_s${SEED}_${TIMESTAMP}.log"

            set +e
            train_one "$algo" "$CONFIG_UPGRADE" "$SEED" "$TRAIN_LOG_XLARGE"
            UPGRADE_EXIT_CODE=$?
            set -e

            if [ $UPGRADE_EXIT_CODE -eq 0 ]; then
                XLARGE_ACC=$(best_acc "$TRAIN_LOG_XLARGE")
                if [ -n "$XLARGE_ACC" ] && gt "$XLARGE_ACC" "$SEED_ACC"; then
                    echo "Upgraded! $CONFIG_UPGRADE acc=$XLARGE_ACC > $SEED_ACC"
                    SEED_ACC=$XLARGE_ACC
                    SEED_CONFIG=$CONFIG_UPGRADE
                else
                    echo "No improvement from upgrade, keeping $CONFIG_INITIAL result."
                fi
            else
                echo "Upgrade training failed, keeping $CONFIG_INITIAL result"
            fi
        fi

        # The best seed's checkpoint stays in place for promotion; every other seed is banked.
        if [ -z "$BEST_ACC" ] || gt "$SEED_ACC" "$BEST_ACC"; then
            [ -n "$BEST_SEED" ] && bank "$algo" "$BEST_SEED"
            BEST_ACC=$SEED_ACC
            FINAL_CONFIG=$SEED_CONFIG
            BEST_SEED=$SEED
        else
            bank "$algo" "$SEED"
        fi
    done

    if [ -z "$BEST_ACC" ]; then
        FAILURES=$((FAILURES + 1))
        echo "RESULT: FAILURE (no usable run across $SEEDS seed(s))"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Final: $algo | acc=$BEST_ACC | config=$FINAL_CONFIG | seed=$BEST_SEED"
    echo "=========================================="

    # Promotion goes through the ONE gate (promote.py): it re-reads the checkpoint's own
    # recorded seq-acc and refuses an unmeasured or sub-gate one. This used to be an
    # unconditional `cp` that ran AFTER printing RESULT: FAILURE, which is how 49 sub-gate
    # checkpoints shipped and 209 circuit rows got built on them.
    if "$PY" promote.py "$CHECKPOINTS_DIR/checkpoint_${algo}_best.pt" \
            "$TRAINED_MODELS_PT/${algo}.pt" --gate "$ACCURACY_THRESHOLD"; then
        echo "$algo  # acc=$BEST_ACC config=$FINAL_CONFIG seed=$BEST_SEED" >> "$SUCCESS_FILE"
        SUCCESSES=$((SUCCESSES + 1))
        echo "RESULT: SUCCESS (>= $ACCURACY_THRESHOLD, promoted)"
    else
        echo "$algo  # acc=$BEST_ACC config=$FINAL_CONFIG seed=$BEST_SEED" >> "$FAILURE_FILE"
        FAILURES=$((FAILURES + 1))
        echo "RESULT: FAILURE (< $ACCURACY_THRESHOLD, NOT promoted)"
    fi
done

echo ""
echo "=========================================="
echo "Training Job Complete"
echo "=========================================="
echo "Total: $TOTAL"
echo "Successes (>= $ACCURACY_THRESHOLD, promoted): $SUCCESSES"
echo "Failures (< $ACCURACY_THRESHOLD, not promoted): $FAILURES"
echo "Upgrades attempted (>= $UPGRADE_THRESHOLD): $UPGRADES"
echo ""
echo "Success file: $SUCCESS_FILE"
echo "Failure file: $FAILURE_FILE"
echo "Logs: $JOBS_DIR/train_*_${TIMESTAMP}.log"

# post-train gate: any algorithm below the success threshold fails the job (blocks analysis)
if [ "$FAILURES" -gt 0 ]; then
    echo "POST-TRAIN GATE: $FAILURES algorithm(s) below $ACCURACY_THRESHOLD -> exit 1"
    exit 1
fi

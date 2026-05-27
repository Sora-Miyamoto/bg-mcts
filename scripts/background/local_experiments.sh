#!/bin/bash
echo "🏃 experiments Start 🏃"

CONFIG_NAME=$1
EXP_ID=$2

# limit setting
LIMIT_RULE=$3

# Algorithsms you use
ALGO_CLASS_NAME=$4
STOPPING_RULE=$5
DIST_TYPE=$6

# Number of parallel jobs
N_JOBS=$7

#  Used Model
MODEL=$8

# Task Setting
TASK=$9
INDICES_FILE=${10}

# Vllm PID
VLLM_PID0=${11}
VLLM_PID1=${12}
MAIN_VENV=${13}
GEN_VENV=${14}
PRM_VENV=${15}
# Track execution time
start_time=$(date +%s)

cleanup() {
    local temp_env_root=""

    if [ -n "$VLLM_PID0" ]; then
        kill "$VLLM_PID0" 2>/dev/null || true
    fi

    if [ -n "$VLLM_PID1" ]; then
        kill "$VLLM_PID1" 2>/dev/null || true
    else
        echo "Gen PRM server not exists"
    fi

    if [ -n "$MAIN_VENV" ]; then
        temp_env_root=$(dirname "$MAIN_VENV")
        rm -rf "$MAIN_VENV"
    fi

    if [ -n "$GEN_VENV" ]; then
        rm -rf "$GEN_VENV"
    fi

    if [ -n "$PRM_VENV" ]; then
        rm -rf "$PRM_VENV"
    fi

    if [ -n "$temp_env_root" ]; then
        rm -rf "$temp_env_root"
    fi
}

trap cleanup EXIT

source .env
export FOLDER_PATH
export HF_TOKEN
RESULTS_FOLDER="${FOLDER_PATH}outputs/${TASK}/${EXP_ID}/${MODEL}/${ALGO_CLASS_NAME}_${STOPPING_RULE}/results"
mkdir -p "$RESULTS_FOLDER"

# Execute tasks in parallel using GNU parallel
# Each task ID from the indices file will be processed concurrently

source "${MAIN_VENV}/bin/activate"
cd ${FOLDER_PATH}

echo "🚀 Starting parallel execution with $N_JOBS jobs..."
cat $INDICES_FILE | parallel -j $N_JOBS "
    # Set task-specific variables
    export TASK_ID="{}"
    CKPT_PATH=outputs/${TASK}/${EXP_ID}/${MODEL}/${ALGO_CLASS_NAME}_${STOPPING_RULE}/results/{}/checkpoints/checkpoint_latest.pkl
    
    # Create output directory for this task
    mkdir -p ${RESULTS_FOLDER}/{}
    
    # Build the command to run
    CMD=\"${MAIN_VENV}/bin/python experiments/math/run_vllm.py \\
        --config-path ${FOLDER_PATH}outputs/${TASK}/${EXP_ID}/${MODEL}/${ALGO_CLASS_NAME}_${STOPPING_RULE}/ \\
        --config-name ${CONFIG_NAME} \\
        task_setting.task_id=\\\"{}\\\" \\
        +tree_photo_path=${FOLDER_PATH}outputs/${TASK}/${EXP_ID}/${MODEL}/${ALGO_CLASS_NAME}_${STOPPING_RULE}/picture \\
        ++hydra.run.dir=${RESULTS_FOLDER}/{}\"
    
    # Add checkpoint path if it exists
    if [ -e \"\$CKPT_PATH\" ]; then
        CMD=\"\$CMD checkpoint_path=\$CKPT_PATH\"
    fi
    
    echo 'Starting task: {}'
    eval \"\$CMD\"
"

# Calculate and display execution time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
elapsed_minutes=$(awk "BEGIN {printf \"%.2f\", $elapsed_time/60}")

echo "✅All tasks completed. Total time: ${elapsed_minutes} minutes."

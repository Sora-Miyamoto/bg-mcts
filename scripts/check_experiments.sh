#!/bin/bash

# how to use
# bash scripts/check_experiments.sh

set -euo pipefail

source .env

if [ -z "${FOLDER_PATH:-}" ]; then
    echo "❌ FOLDER_PATH is not set in .env"
    exit 1
fi

################ ⚙️ Script Setting ⚙️ ######################
EXP_ID=test/1   
STOPPING_RULE=sequential    # "full": For full generation algorithm. "sequential": For Sequential generation algorithm.
ALGO_CLASS_NAME="LiteSearchBatch"   # Algorithms you use. We provide some algorrithms below.
# "SequentialRefinement", "RepeatedSampling", "ABMCTSA", "ABMCTSM", "StandardMCTS", 
# "LiteSearchBatch", "LiteSearchIncremental","BGMCTS", "TreeOfThoughtsBFSAlgo"

TASK=Math500   #you can set Math500, Minervamath and AIME24_25 right now

# Running Task problem ids. You can set the txt file with task ids
INDICES_FILE="${FOLDER_PATH}benchmarks/Math500/math500_full.txt" 
# INDICES_FILE="${FOLDER_PATH}benchmarks/AIME24_25/aime24_25_short.txt"
# INDICES_FILE="${FOLDER_PATH}benchmarks/Minervamath/minervamath_short.txt"

# Model Setting. You can set the model you want to check.
MODEL="Qwen/Qwen3-14B"
# "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",  
# "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"
# "google/gemma-3-4b-it", "google/gemma-3-12b-it", 

###############################################################

EXPERIMENT_FOLDER="${FOLDER_PATH}outputs/${TASK}/${EXP_ID}/${MODEL}/${ALGO_CLASS_NAME}_${STOPPING_RULE}"

TMP_ENV_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/check-XXXXXX")
ANALYSIS_VENV="${TMP_ENV_ROOT}/check"

cleanup_temp_env() {
    if [ -n "${TMP_ENV_ROOT:-}" ] && [ -d "${TMP_ENV_ROOT}" ]; then
        rm -rf "${TMP_ENV_ROOT}"
    fi
}

trap cleanup_temp_env EXIT

create_temp_uv_env() {
    local env_path=$1
    local requirements_file=$2

    echo "🐍 Creating temporary uv environment: ${env_path}"
    if [ ! -f "${requirements_file}" ]; then
        echo "❌ Requirements file not found: ${requirements_file}"
        exit 1
    fi

    uv venv "${env_path}" || exit 1
    source "${env_path}/bin/activate"
    uv pip sync "${requirements_file}" || exit 1
}

create_temp_uv_env "${ANALYSIS_VENV}" "${FOLDER_PATH}requirements/requirements_analysis.txt"

source "${ANALYSIS_VENV}/bin/activate"
cd "${FOLDER_PATH}"

${ANALYSIS_VENV}/bin/python eval/check_experiment_progress.py \
    --exp_path=${EXPERIMENT_FOLDER} \
    --task_file=${INDICES_FILE} \

echo "✅ Experiment check completed for ${EXPERIMENT_FOLDER}"

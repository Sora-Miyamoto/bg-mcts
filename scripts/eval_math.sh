#!/bin/bash
# how to use
# bash scripts/eval_math.sh

source .env

# ===================== ⚙️ Setting ⚙️ =====================

# set the result path
exp_path_list=(

)

#  set the graph  name
graph_name_list=(

)

############## Example ####################
# You can set the path of result you want to see the results. The path should be like this: 
# exp_path_list=(
#     ${FOLDER_PATH}outputs/Math500/${EXP_ID}/Qwen/Qwen2.5-7B-Instruct/ABMCTSM_full
#     ${FOLDER_PATH}outputs/Math500/${EXP_ID}/Qwen/Qwen2.5-7B-Instruct/StandardMCTS_sequential
#     ${FOLDER_PATH}outputs/Math500/${EXP_ID}/Qwen/Qwen2.5-7B-Instruct/BGMCTS_sequential
# )

# Set the graph  name
# graph_name_list=(
#     "ABMCTSM"
#     "StandardMCTS"
#     "BGMCTS"
# )
############################################

max_cost="20000"  # The given budget for each tree search.
proc=True # Whether to process the results before visualization. If you have already processed the results, you can set it to False to save time.

# set Graph path
file_name="Eval_Results"
path_to_graph="${FOLDER_PATH}outputs/Math500/Plots/"

# ---------------🎓 Set the path to math tasks 🎓---------------------

# --------------- MATH500 ---------------
path_to_math_tasks="${FOLDER_PATH}benchmarks/Math500/math500_level5.txt"
# You can set the txt files below.
# "math500_short.txt", "math500_full.txt", "math500_level1.txt", "math500_level2.txt", "math500_level3.txt", "math500_level4.txt", "math500_level5.txt"

# --------------- AIME24/25 ---------------
# path_to_math_tasks="${FOLDER_PATH}benchmarks/AIME24/25/aime24_25_short.txt"
# You can set the txt files below.
# "aime24_25_short.txt", "aime24_25_full.txt"

# --------------- Minervamath ---------------
# path_to_math_tasks="${FOLDER_PATH}benchmarks/Minervamath/minervamath_short.txt"
# You can set the txt files below.
# "minervamath_short.txt", "minervamath_full.txt"

# ================================================================

mkdir -p "$path_to_graph"
path_to_ckpt="{exp_name}/results/{task_id}/checkpoints/checkpoint_total.pkl"

TMP_ENV_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/evaluation-XXXXXX")
ANALYSIS_VENV="${TMP_ENV_ROOT}/check"

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
cd ${FOLDER_PATH}

if [[ $proc == "True" ]]; then
    for exp_name in ${exp_path_list[@]}; do
        DATA_DIR=${exp_name}/data
        mkdir -p "$DATA_DIR"
        ${ANALYSIS_VENV}/bin/python eval/proc_results.py \
            --exp_name=${exp_name} \
            --path_to_math_tasks=${path_to_math_tasks} \
            --path_to_ckpt=${path_to_ckpt} \
            --n_jobs=10
    done
fi

exp_paths_str=$(IFS=','; echo "${exp_path_list[*]}")
graph_names_str=$(IFS=','; echo "${graph_name_list[*]}")

export exp_paths_str 
export graph_names_str 

${ANALYSIS_VENV}/bin/python eval/visualize.py \
    --max_cost=${max_cost} \
    --pic_name=${file_name} \
    --path_to_graph=${path_to_graph} \


if [ -n "$ANALYSIS_VENV" ]; then
    temp_env_root=$(dirname "$ANALYSIS_VENV")
    rm -rf "$ANALYSIS_VENV"
fi
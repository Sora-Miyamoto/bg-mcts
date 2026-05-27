#!/bin/bash

# how to use
# bash scripts/run_local_template.sh job1 
# log is saved in logs/job1.log

source .env
######################## ⚙️ Setting You need to write down ⚙️########################

# ===========🔧 Base Setting 🔧 ===========
ENVIRONMENT=local
CONFIG_NAME=config.yaml
EXP_ID=test/   

# limit setting
LIMIT_RULE="Cost_Base"      # "Node_Base" or "Cost_Base"
MAX_NUM_NODES=1             # Node
MAX_NUM_COST=20000          # Outputs token budget in Search Algorithm
STOPPING_RULE=sequential    # "full": For full generation algorithm. "sequential": For Sequential generation algorithm.
# =========================================

# ===============🎲　Algorithms Setting 🎲===============

ALGO_CLASS_NAME="BGMCTS"   # Algorithms you use. We provide some algorrithms below.
# "SequentialRefinement", "RepeatedSampling", "ABMCTSA", "ABMCTSM", "StandardMCTS", 
# "LiteSearchBatch", "LiteSearchIncremental","BGMCTS", "TreeOfThoughtsBFSAlgo"

#  ------- 🎛️ Algorithm parameters setting 🎛️ --------
#  MCTS and BGMCTS parameters:
SAMPLES_PER_ACTION=2  # Number of samples to generate for each action. (e.g., 1, 3, 5, 10)
EXPLORATION_WEIGHT=2  # c: The weight of the exploration term.  ⚠️This number goes inside the square root.
DEPTH_WEIGHT=1        # κ: The weight of the completion bias in Exploitation term used in BG-PUCT. (e.g., 0.5, 1)
VARUNCE_WEIGHT=1      # λ: The weight of the variance in generative node score. (e.g., 0.5, 1)
BUDGET="${MAX_NUM_COST}"          # B: The number of given budget. (e.g., 10000, 20000, 30000)


# LiteSearch parameters:
MAX_EXPANSION=10      # B: The maximum number of expansions for each node.
LAMBDA=0              # λ: The weight of the step score.
EPSILON=0.8           # ε: The threshold for the stopping rule.

# TreeOfThoughtsBFSAlgo parameters:
BREADH_LIMIT=5       # b: The number of top-scoring nodes selected from the deepest level at each step.
SIZE_LIMIT=10        # k: The total number of samples generated per selected node, split across actions.
# -------------------------------------------------------
# =========================================================

# ----------🏋️‍♀️ Number of parallel jobs 🏋️‍♀️----------
N_JOBS=10

# ----------📸 Tree photo setting 📸--------------
# Whether to visualize the search tree. If True, the search tree will be visualized and saved in the output directory.
# A visualized tree example is here images/bgmcts_tree_example.pdf. 
# ⚠️　If you set True, it may use more memory.
VISUALIZE=True

# =================🎓 Task Setting 🎓=================
# 📝 What tasks you use
# --------------- MATH500 ---------------
TASK=Math500   #you can set Math500, Minervamath and AIME24_25 right now
INDICES_FILE="${FOLDER_PATH}benchmarks/Math500/math500_level5.txt"  # Task problem ids to run. You can set the txt file with task ids
# You can set the txt files below.
# "math500_short.txt", "math500_full.txt", "math500_level1.txt", "math500_level2.txt", "math500_level3.txt", "math500_level4.txt", "math500_level5.txt"

# --------------- AIME24/25 ---------------
# TASK=AIME24_25   #you can set, Math500, Minervamath and AIME24_25 right now
# INDICES_FILE="${FOLDER_PATH}benchmarks/AIME24_25/aime24_25_short.txt" # Task problem ids to run. You can set the txt file with task ids
# You can set the txt files below.
# "aime24_25_short.txt", "aime24_25_full.txt"

# --------------- Minervamath ---------------
# TASK=Minervamath   #you can set Math500, Minervamath and AIME24_25 right now
# INDICES_FILE="${FOLDER_PATH}benchmarks/Minervamath/minervamath_short.txt"  # Task problem ids to run. You can set the txt file with task ids
# You can set the txt files below.
# "minervamath_short.txt", "minervamath_full.txt"
# ===============================================================

# ====================🏎️ Model Setting 🏎️=====================

# ----------🧠 Models 🧠----------
# What generation model you use
MODEL="Qwen/Qwen2.5-7B-Instruct"
# "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",  
# "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"
# "google/gemma-3-4b-it", "google/gemma-3-12b-it", 


# What PRM you use 
PRM="GenPRM/GenPRM-7B" # You can set "GenPRM/GenPRM-7B"

# ----------⚙️ Generation Model Setting ⚙️----------
GPU_INDICES=0            # The index of GPU for generation model.  If you have GPU 0 you can set "0".
GPU_NUM=1                # Number of GPUs you use (e.g., 1, 2, 4, 8)
GPU_UTIL=0.95            # GPU memory utilization (e.g., 0.60, 0.95, 0.90)
HOST=0.0.0.0             # Host for VLLM server.
PORT=10003               # Port for VLLM server.
DTYPE=bfloat16           # Dtype for VLLM. 
WHERE_CUT="middle"       # you can set "middle"
MAX_MODEL_LEN=16384      # max tokens that llm can treat (InputTokens + OutputTokens)
MAX_TOKENS=8192          # Max tokens llm will generate(OutputTokens)
TEMPERATURE=0.6          # Temperature for sampling. (e.g., 0.6, 0.8, 1.0)
TOP_P=1.0                # Top-p (nucleus) sampling. (e.g., 0.8, 0.9, 0.95, 1.0)
REPETITION_PENALTY=1.0   # Repetition penalty. (e.g., 1.0, 1.2, 1.5)

# ----------⚙️ GenPRM Setting ⚙️----------
PRM_GPU_INDICES=1             # The index of GPU for GenPRM. If you have GPU 1,2 you can set "1,2".
PRM_GPU_NUM=1                 # Number of GPUs you use for GenPRM (e.g., 1, 2, 4, 8)
PRM_GPU_UTIL=0.95             # GPU memory utilization for GenPRM (e.g., 0.60, 0.95, 0.90)
PRM_HOST=0.0.0.0              # Host for GenPRM VLLM server. 
PRM_PORT=10004                # Port for GenPRM VLLM server. 
PRM_DTYPE=bfloat16            # Dtype for GenPRM.
PRM_MAX_MODEL_LEN=32768       # max tokens that GenPRM can treat (InputTokens + OutputTokens)
PRM_MAX_TOKENS=1024           # Max tokens GenPRM will generate(OutputTokens)
PRM_TEMPERATURE=0.60          # Temperature for sampling in GenPRM. (e.g., 0.6, 0.8, 1.0)
PRM_TOP_P=0.95                # Top-p (nucleus) sampling for GenPRM. (e.g., 0.8, 0.9, 0.95, 1.0)
PRM_TOP_K=20                  # Top-k sampling for GenPRM. (e.g., 10, 20, 50, 100)
PRM_TOP_LOGPROBS=20           # The number of top logprobs to return for GenPRM. (e.g., 10, 20, 50, 100) 
PRM_REPETITION_PENALTY=1.0    # Repetition penalty for GenPRM. (e.g., 1.0, 1.2, 1.5)
PRM_VLLM_SEED=1               # Seed for GenPRM VLLM server. You can set any integer (e.g., 1, 42, 100).               
ANALYZE=True                  # Whether to analyze the generation 
VERIFY=False                  # Whether to verify the generation in 
EXECUTE=False                 # Whether to execute the generated code.
TIME_LIMIT=3                  # Time limit for each PRM call in seconds. 
SCORING="last"                # "average" or "last" or "min" when you use full generartion algorithm.
PRM_CALL=1                    # How many times you call PRM
# ========================================================================================================

########################################################################################################################

# ----------📝 Writing config file 📝----------
CONFIG_FILE="${FOLDER_PATH}outputs/${TASK}/${EXP_ID}/${MODEL}/${ALGO_CLASS_NAME}_${STOPPING_RULE}/config.yaml"
mkdir -p "$(dirname "${CONFIG_FILE}")"
cat > "${CONFIG_FILE}" << YAML
environment: ${ENVIRONMENT}

task_setting: 
  task_name: ${TASK}
  task_id: !!str null
  indices_file: ${INDICES_FILE}

prompt_version: null

limit_setting: 
  limit_rule: ${LIMIT_RULE}
  max_num_nodes: ${MAX_NUM_NODES}
  max_num_cost: ${MAX_NUM_COST}
  stopping_rule: ${STOPPING_RULE}

algo:
  class_name: ${ALGO_CLASS_NAME}
  params:
    model_selection_strategy: "stack"
    samples_per_action: ${SAMPLES_PER_ACTION}
    exploration_weight: ${EXPLORATION_WEIGHT}
    depth_weight: ${DEPTH_WEIGHT}
    variance_weight: ${VARUNCE_WEIGHT}
    budget: ${BUDGET}
    max_expansion: ${MAX_EXPANSION} 
    lambda: ${LAMBDA}
    epsilon: ${EPSILON}
    breadth_limit: ${BREADH_LIMIT}
    size_limit: ${SIZE_LIMIT}

n_jobs: ${N_JOBS}

visualize: ${VISUALIZE}

checkpoint_path: null

models:
  - name: ${MODEL}
    temperature: ${TEMPERATURE}

prms: 
  - name: ${PRM}

vllm_setting:
  gpu: ${PRM_GPU_INDICES}
  gpu_num: ${GPU_NUM}
  gpu_util: ${GPU_UTIL}
  server_host: ${HOST}
  server_port: ${PORT}
  server_dtype: ${DTYPE}
  where_cut: ${WHERE_CUT}
  max_model_len: ${MAX_MODEL_LEN}
  max_tokens: ${MAX_TOKENS}
  temperature: ${TEMPERATURE}
  top_p: ${TOP_P}
  repetition_penalty: ${REPETITION_PENALTY}

gen_prm_setting:
  gpu: ${PRM_GPU_INDICES}
  gpu_num: ${PRM_GPU_NUM}
  gpu_util: ${PRM_GPU_UTIL}
  server_host: ${PRM_HOST}
  server_port: ${PRM_PORT}
  server_dtype: ${PRM_DTYPE}
  max_model_len: ${PRM_MAX_MODEL_LEN}
  max_tokens: ${PRM_MAX_TOKENS}
  temperature: ${PRM_TEMPERATURE}
  top_p: ${PRM_TOP_P}
  top_k: ${PRM_TOP_K}
  top_logprobs: ${PRM_TOP_LOGPROBS}
  repetition_penalty: ${PRM_REPETITION_PENALTY}
  vllm_seed: ${PRM_VLLM_SEED}
  analyze: ${ANALYZE}
  verify: ${VERIFY}
  execute: ${EXECUTE}
  time_limit: ${TIME_LIMIT}
  scoring: ${SCORING}
  prm_call: ${PRM_CALL}


YAML
echo "📝  Wrote ${CONFIG_FILE} 📝"


# ジョブ名を引数で受け取る
JOB_NAME=$1
if [ -z "$JOB_NAME" ]; then
    echo "使い方: $0 <job_name>"
    exit 1
fi

# ログ保存先
LOG_DIR="${FOLDER_PATH}logs"
mkdir -p "$LOG_DIR"
LOG_FOLDER="${FOLDER_PATH}logs/${JOB_NAME}"
mkdir -p "$LOG_FOLDER"
LOG_FILE="${LOG_FOLDER}/${JOB_NAME}.log"
LLM_LOG_FILE="${LOG_FOLDER}/vllm_server.log"
PRM_LOG_FILE="${LOG_FOLDER}/prm_vllm_server.log"

TMP_ENV_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/bg-mcts-${JOB_NAME}-XXXXXX")
MAIN_VENV="${TMP_ENV_ROOT}/main"
GEN_VENV="${TMP_ENV_ROOT}/gen"
PRM_VENV="${TMP_ENV_ROOT}/prm"
HANDOFF_CLEANUP=0

cleanup_temp_envs() {
    if [ "${HANDOFF_CLEANUP}" -eq 1 ]; then
        return
    fi

    rm -rf "${TMP_ENV_ROOT}"
}

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

trap cleanup_temp_envs EXIT

create_temp_uv_env "${GEN_VENV}" "${FOLDER_PATH}requirements/requirements_gen.txt"


echo "📟 Start setting up VLLM Server 📟"

if [[ $MODEL == "meta-llama/Llama-3.1-8B-Instruct" || $MODEL == "meta-llama/Llama-3.2-3B-Instruct" ]]; then
    CHAT_TEMPLATE=${FOLDER_PATH}src/bg_mcts/llm/vllm/llama_chat_template.jinja
elif [[ $MODEL == "Qwen/Qwen2.5-7B-Instruct" ]]; then
    CHAT_TEMPLATE=${FOLDER_PATH}src/bg_mcts/llm/vllm/qwen2_chat_template.jinja
elif [[ $MODEL == "Qwen/Qwen3-8B" ]]; then
    CHAT_TEMPLATE=${FOLDER_PATH}src/bg_mcts/llm/vllm/qwen3_chat_template.jinja
elif [[ $MODEL == "Qwen/Qwen3-14B" ]]; then
    CHAT_TEMPLATE=${FOLDER_PATH}src/bg_mcts/llm/vllm/qwen3_chat_template.jinja
elif [[ $MODEL == "Qwen/Qwen3-32B" ]]; then
    CHAT_TEMPLATE=${FOLDER_PATH}src/bg_mcts/llm/vllm/qwen3_chat_template.jinja
elif [[ $MODEL == "google/gemma-3-4b-it" || $MODEL == "google/gemma-3-12b-it" ]]; then
    CHAT_TEMPLATE=${FOLDER_PATH}src/bg_mcts/llm/vllm/gemma_chat_template.jinja
fi

echo "CHAT_TEMPLATE=$CHAT_TEMPLATE"
CUDA_VISIBLE_DEVICES=$GPU_INDICES \
vllm serve $MODEL \
    --host $HOST \
    --port $PORT \
    --chat-template "$CHAT_TEMPLATE" \
    --hf-token $HF_TOKEN \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --dtype $DTYPE \
    --tensor-parallel-size $GPU_NUM \
    > "$LLM_LOG_FILE" 2>&1 &

VLLM_PID0=$!
echo "VLLM server started with PID: $VLLM_PID0"
sleep 10
ps -fp $VLLM_PID0


TARGET_SCRIPT="${FOLDER_PATH}scripts/background/local_experiments.sh"

if [[ $PRM == "GenPRM/GenPRM-7B" ]]; then
    echo "📟 Start setting up GenPRM VLLM Server 📟"
    create_temp_uv_env "${PRM_VENV}" "${FOLDER_PATH}requirements/requirements_prm.txt"
    CUDA_VISIBLE_DEVICES=$PRM_GPU_INDICES \
    vllm serve "GenPRM/GenPRM-7B" \
        --host $PRM_HOST \
        --port $PRM_PORT \
        --hf-token $HF_TOKEN \
        --max-model-len $PRM_MAX_MODEL_LEN \
        --gpu-memory-utilization $PRM_GPU_UTIL \
        --dtype $PRM_DTYPE \
        --tensor-parallel-size $PRM_GPU_NUM \
        > "$PRM_LOG_FILE" 2>&1 &
        VLLM_PID1=$!
        echo "VLLM server started with PID: $VLLM_PID1"
        sleep 90
        ps -fp $VLLM_PID1

    create_temp_uv_env "${MAIN_VENV}" "${FOLDER_PATH}requirements/requirements.txt"
    nohup bash "$TARGET_SCRIPT" \
        "$CONFIG_NAME" \
        "$EXP_ID" \
        "$LIMIT_RULE" \
        "$ALGO_CLASS_NAME" \
        "$STOPPING_RULE" \
        "$DIST_TYPE" \
        "$N_JOBS" \
        "$MODEL" \
        "$TASK" \
        "$INDICES_FILE" \
        "$VLLM_PID0" \
        "$VLLM_PID1" \
        "$MAIN_VENV" \
        "$GEN_VENV" \
        "$PRM_VENV" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
else
    create_temp_uv_env "${MAIN_VENV}" "${FOLDER_PATH}requirements/requirements.txt"
    nohup bash "$TARGET_SCRIPT" \
        "$CONFIG_NAME" \
        "$EXP_ID" \
        "$LIMIT_RULE" \
        "$ALGO_CLASS_NAME" \
        "$STOPPING_RULE" \
        "$DIST_TYPE" \
        "$N_JOBS" \
        "$MODEL" \
        "$TASK" \
        "$INDICES_FILE" \
        "$VLLM_PID0" \
        "$MAIN_VENV" \
        "$GEN_VENV" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
fi

HANDOFF_CLEANUP=1
echo "🚀 $TARGET_SCRIPT has started (JOB: $JOB_NAME, PID: $PID) 🚀"

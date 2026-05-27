import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from fire import Fire
from joblib import Parallel, delayed
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT / "src", REPO_ROOT / "treesearch" / "src", REPO_ROOT / "experiments" / "math"):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.append(extra_path_str)

def get_prm_score_total(node):
    scores = [node.score]

    node = node.parent
    while node is not None:
        if node.state is None or node.state.generation_result is None:
            break
        if node.state.generation_result.generation_state[0] != "process":
            break
        scores.append(node.score)
        node = node.parent
        # print(scores)
    return float(sum(scores) / len(scores))

def get_prm_score_min(node):
    scores = [node.score]

    node = node.parent
    while node is not None:
        if node.state is None or node.state.generation_result is None:
            break
        if node.state.generation_result.generation_state[0] != "process":
            break
        scores.append(node.score)
        node = node.parent
        # print(scores)
    min_score = min(scores)
    return min_score

def process_node(node):
    """Helper function to process a single node and calculate scores."""
    if node.expand_idx < 0:
        return None
    # if node.state == None or node == None:
    #     return None
    node_idx = node.expand_idx
    node_state = node.state.generation_result.generation_state[0]
    prm_score = node.score
    prm_score_min = get_prm_score_min(node)
    prm_score_total = get_prm_score_total(node)
    node_cost = node.cost
    node_total_cost = node.total_cost
    node_output_cost = node.output_cost
    node_total_output_cost = node.total_output_cost
    node_input_cost = node.input_cost
    node_total_input_cost = node.total_input_cost
    node_prm_output_cost = node.prm_output_cost
    node_prm_total_output_cost = node.prm_total_output_cost
    node_prm_input_cost = node.prm_input_cost
    node_prm_total_input_cost = node.prm_total_input_cost
    node_depth = node.depth
    

    return node_idx, node_state, prm_score, prm_score_min, prm_score_total, node_cost, node_total_cost, node_output_cost, node_total_output_cost, node_input_cost, node_total_input_cost, node_prm_output_cost, node_prm_total_output_cost, node_prm_input_cost, node_prm_total_input_cost, node_depth

def main(
    exp_name: str,
    n_jobs: int = 4,
    path_to_ckpt: str = "{exp_name}/{task_id}/checkpoints/checkpoint_n_answers_128.pkl",
    save_path: str = "{exp_name}/data",
    path_to_math_tasks: str = "./experiments/math/Math500_short.txt",
):
    output_dir = Path(save_path.format(exp_name=exp_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(path_to_math_tasks, "r") as f:
        task_list = f.readlines()
    task_list = [t.strip() for t in task_list]

    node_idx_dict = {}
    node_state_dict = {}
    prm_score_dict = {}
    prm_score_min_dict = {}
    prm_score_total_dict = {}
    node_cost_dict = {}
    node_total_cost_dict = {}
    node_output_cost_dict = {}
    node_total_output_cost_dict = {} 
    node_input_cost_dict =  {}
    node_total_input_cost_dict = {}
    node_prm_output_cost_dict = {} 
    node_prm_total_output_cost_dict = {} 
    node_prm_input_cost_dict = {} 
    node_prm_total_input_cost_dict = {}
    node_depth_dict = {}

    nonexistent_task = []

    for task_id in task_list:
        math_problem_path = Path(f"./Math500/info/{task_id}.json")

        # load state
        state_path = Path(path_to_ckpt.format(exp_name=exp_name, task_id=task_id))
        proc_ret_path = Path(state_path.as_posix().replace(".pkl", "_proc_result.json"))
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        # Filter nodes with expand_idx >= 0 before processing
        valid_nodes = [
            node for node in state.tree.get_nodes() if node.expand_idx >= 0
        ]

        # Parallel processing of nodes
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_node)(node)
            for node in tqdm(valid_nodes, desc=f"Processing {task_id}")
        )

        # Filter out None results (from nodes with expand_idx < 0, though already filtered)
        results = [r for r in results if r is not None]

        # Sort results by node_idx (the first element of each tuple in results)
        results.sort(key=lambda x: x[0])

        # Unpack sorted results
        node_idx_list = []
        node_state_list = []
        prm_score_list = []
        prm_score_min_list = []
        prm_score_total_list = []
        node_cost_list =  []
        node_total_cost_list = []
        node_output_cost_list = []
        node_total_output_cost_list = []
        node_input_cost_list = []
        node_total_input_cost_list = []
        node_prm_output_cost_list = []
        node_prm_total_output_cost_list = []
        node_prm_input_cost_list = []
        node_prm_total_input_cost_list = []
        node_depth_list = []

        for result in results:
            node_idx, node_state, prm_score, prm_score_min, prm_score_total, node_cost, node_total_cost, node_output_cost, node_total_output_cost, node_input_cost, node_total_input_cost, node_prm_output_cost, node_prm_total_output_cost, node_prm_input_cost, node_prm_total_input_cost, node_depth = (
                result  # No need to check for None here, already filtered
            )
            node_idx_list.append(node_idx)
            node_state_list.append(node_state)
            prm_score_list.append(prm_score)
            prm_score_min_list.append(prm_score_min)
            prm_score_total_list.append(prm_score_total)
            node_cost_list.append(node_cost)
            node_total_cost_list.append(node_total_cost)
            node_output_cost_list.append(node_output_cost)
            node_total_output_cost_list.append(node_total_output_cost)
            node_input_cost_list.append(node_input_cost)
            node_total_input_cost_list.append(node_total_input_cost)
            node_prm_output_cost_list.append(node_prm_output_cost)
            node_prm_total_output_cost_list.append(node_prm_total_output_cost)
            node_prm_input_cost_list.append(node_prm_input_cost)
            node_prm_total_input_cost_list.append(node_prm_total_input_cost)
            node_depth_list.append(node_depth)

        proc_ret = {
            "node_idx_list": node_idx_list,
            "node_state_list": node_state_list,
            "prm_score_list": prm_score_list,
            "prm_score_min_list": prm_score_min_list,
            "prm_score_total_list": prm_score_total_list,
            "node_cost_list": node_cost_list,
            "node_total_cost_list": node_total_cost_list,
            "node_output_cost_list": node_output_cost_list,
            "node_total_output_cost_list": node_total_output_cost_list,
            "node_input_cost_list": node_input_cost_list,
            "node_total_input_cost_list": node_total_input_cost_list,
            "node_prm_output_cost_list": node_prm_output_cost_list,
            "node_prm_total_output_cost_list": node_prm_total_output_cost_list,
            "node_prm_input_cost_list": node_prm_input_cost_list,
            "node_prm_total_input_cost_list": node_prm_total_input_cost_list,
            "node_depth_list": node_depth_list,
            "nonexistent_task": nonexistent_task,
        }
        with open(proc_ret_path, "w") as f:
            json.dump(proc_ret, f)

        node_idx_dict[task_id] = node_idx_list
        node_state_dict[task_id] = node_state_list
        prm_score_dict[task_id] = prm_score_list
        prm_score_min_dict[task_id] = prm_score_min_list
        prm_score_total_dict[task_id] = prm_score_total_list
        node_cost_dict[task_id] = node_cost_list
        node_total_cost_dict[task_id] = node_total_cost_list
        node_output_cost_dict[task_id] = node_output_cost_list
        node_total_output_cost_dict[task_id] = node_total_output_cost_list
        node_input_cost_dict[task_id] = node_input_cost_list
        node_total_input_cost_dict[task_id] = node_total_input_cost_list
        node_prm_output_cost_dict[task_id] = node_prm_output_cost_list
        node_prm_total_output_cost_dict[task_id] = node_prm_total_output_cost_list
        node_prm_input_cost_dict[task_id] = node_prm_input_cost_list
        node_prm_total_input_cost_dict[task_id] = node_prm_total_input_cost_list
        node_depth_dict[task_id] = node_depth_list
    
    max_len = max(len(v) for v in node_idx_dict.values())
    for k, v in node_idx_dict.items():
        if len(v) < max_len:
            node_idx_dict[k] = v + [None] * (max_len - len(v))
            node_state_dict[k] += [None] * (max_len - len(v))
            prm_score_dict[k] += [None] * (max_len - len(v))
            prm_score_min_dict[k] += [None] * (max_len - len(v))
            prm_score_total_dict[k] += [None] * (max_len - len(v))
            node_cost_dict[k] += [None] * (max_len - len(v))
            node_total_cost_dict[k] += [None] * (max_len - len(v))
            node_output_cost_dict[k] += [None] * (max_len - len(v))
            node_total_output_cost_dict[k] += [None] * (max_len - len(v))
            node_input_cost_dict[k] += [None] * (max_len - len(v))
            node_total_input_cost_dict[k] += [None] * (max_len - len(v))
            node_prm_output_cost_dict[k] += [None] * (max_len - len(v))
            node_prm_total_output_cost_dict[k] += [None] * (max_len - len(v))
            node_prm_input_cost_dict[k] += [None] * (max_len - len(v))
            node_prm_total_input_cost_dict[k] += [None] * (max_len - len(v))
            node_depth_dict[k] += [None] * (max_len - len(v))

    df_node_index = pd.DataFrame(node_idx_dict)
    df_node_state = pd.DataFrame(node_state_dict)
    df_prm_score = pd.DataFrame(prm_score_dict)
    df_prm_score_min = pd.DataFrame(prm_score_min_dict)
    df_prm_score_total = pd.DataFrame(prm_score_total_dict)
    df_cost = pd.DataFrame(node_cost_dict)
    df_total_cost = pd.DataFrame(node_total_cost_dict)
    df_node_output_cost = pd.DataFrame(node_output_cost_dict)
    df_node_total_output_cost = pd.DataFrame(node_total_output_cost_dict)
    df_node_input_cost = pd.DataFrame(node_input_cost_dict)
    df_node_total_input_cost = pd.DataFrame(node_total_input_cost_dict)
    df_node_prm_output_cost = pd.DataFrame(node_prm_output_cost_dict)
    df_node_prm_total_output_cost = pd.DataFrame(node_prm_total_output_cost_dict)
    df_node_prm_input_cost = pd.DataFrame(node_prm_input_cost_dict)
    df_node_prm_total_input_cost = pd.DataFrame(node_prm_total_input_cost_dict)
    df_node_depth = pd.DataFrame(node_depth_dict)

    df_node_index.to_csv(output_dir / "df_node_index.csv", index=False)
    df_node_state.to_csv(
        output_dir / "df_node_state.csv", index=False
    )
    df_prm_score.to_csv(
        output_dir / "df_prm_score.csv", index=False
    )
    df_prm_score_min.to_csv(
        output_dir / "df_prm_score_min.csv", index=False
    )
    df_prm_score_total.to_csv(
        output_dir / "df_prm_score_total.csv", index=False
    )
    df_cost.to_csv(
        output_dir / "df_cost.csv", index=False
    )
    df_total_cost.to_csv(
        output_dir / "df_total_cost.csv", index=False
    )
    df_node_output_cost.to_csv(
        output_dir / "df_node_output_cost.csv", index=False
    )
    df_node_total_output_cost.to_csv(
        output_dir / "df_node_total_output_cost.csv", index=False
    )
    df_node_input_cost.to_csv(
        output_dir / "df_node_input_cost.csv", index=False
    )
    df_node_total_input_cost.to_csv(
        output_dir / "df_node_total_input_cost.csv", index=False
    )
    df_node_prm_output_cost.to_csv(
        output_dir / "df_node_prm_output_cost.csv", index=False
    )
    df_node_prm_total_output_cost.to_csv(
        output_dir / "df_node_prm_total_output_cost.csv", index=False
    )
    df_node_prm_input_cost.to_csv(
        output_dir / "df_node_prm_input_cost.csv", index=False
    )
    df_node_prm_total_input_cost.to_csv(
        output_dir / "df_node_prm_total_input_cost.csv", index=False
    )
    df_node_depth.to_csv(
        output_dir / "df_node_depth.csv", index=False
    )

if __name__ == "__main__":
    Fire(main)

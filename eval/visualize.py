import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire

def get_node_base_data(df_node_state, max_num_nodes):
    array_df_node_state = df_node_state.copy().values
    n_rows, n_cols = array_df_node_state.shape
    node_state_data = np.zeros((max_num_nodes, n_cols))
    for col in range(n_cols):
        correct_idx = np.where(array_df_node_state[:, col] == "correct")[0]
        if correct_idx.size > 0:
            node_state_data[correct_idx[0]:, col] = 1
    return node_state_data


def get_cost_base_data(df_node_state, df_node_total_cost, max_cost):
    array_df_node_state = df_node_state.copy().values
    array_df_node_total_cost = df_node_total_cost.copy().values
    n_rows, n_cols = array_df_node_state.shape
    max_cost_len = max_cost
    node_state_data = np.zeros((max_cost_len, n_cols))
    for col in range(n_cols):
        answered_idx = np.where(array_df_node_state[:, col] != "process")[0]
        if len(answered_idx) == 0:
            continue
        answered_idx_cost = array_df_node_total_cost[answered_idx[0], col]
        if answered_idx_cost <= max_cost:
            node_state_data[int(answered_idx_cost):, col] = 1
    return node_state_data

def get_score_base_data(df_node_state, df_prm_score, df_node_total_cost, max_cost):
    array_df_node_state = df_node_state.copy().values
    array_df_prm_score = df_prm_score.copy().values
    array_df_node_total_cost = df_node_total_cost.copy().values
    n_rows, n_cols = array_df_node_state.shape
    max_cost_len = int(max_cost)
    node_state_data = np.zeros((max_cost_len, n_cols))
    for col in range(n_cols):
        answered_idx = np.where(array_df_node_state[:, col] != "process")[0]
        correct_idx = np.where(array_df_node_state[:, col] == "correct")[0]
        if len(correct_idx) == 0:
            continue
        
        info = {"cost": float(0), "target_score": float(0), "state": "process"}
        for i in range(len(answered_idx)):
            cost = array_df_node_total_cost[answered_idx[i], col]
            if cost > max_cost:
                break
            target_score = array_df_prm_score[answered_idx[i], col]
            if target_score > info["target_score"]:
                info = {
                    "cost": cost,
                    "target_score": target_score,
                    "state": array_df_node_state[answered_idx[i], col]
                }
                if info["state"] == "correct":
                    node_state_data[int(info["cost"]): , col] = 1
                elif info["state"] == "incorrect":
                    node_state_data[int(info["cost"]): , col] = 0
    return node_state_data

def average_num_node(df_node_cost, max_cost):
    array_df_cost = df_node_cost.copy().values
    n_rows, n_cols = array_df_cost.shape
    node_num_sum = 0
    for col in range(n_cols):
        node_index = np.where(array_df_cost[:, col] >= max_cost)[0]
        if len(node_index) == 0:
            node_num_sum += len(array_df_cost[:, col])
        elif array_df_cost[node_index[0], col] == max_cost:
            node_num_sum += node_index[0] + 1
        else:
            node_num_sum += node_index[0]
    
    average_value = node_num_sum / n_cols
    return average_value

def state_num_node(df_node_state, df_node_cost, max_cost):
    array_df_state = df_node_state.copy().values
    array_df_cost = df_node_cost.copy().values
    n_rows, n_cols = array_df_cost.shape
    correct_num = 0
    incorrect_num = 0
    process_num = 0
    non_ans = 0
    correct_ratio = 0
    for col in range(n_cols):
        limit_node_index = np.where(array_df_cost[:, col] >= max_cost)[0]
        if len(limit_node_index) == 0:
            limit_index = len(array_df_cost[:, col])
        elif array_df_cost[limit_node_index[0], col] == max_cost:
            limit_index = limit_node_index[0]
        else:
            limit_index = limit_node_index[0] - 1
        correct_index = np.where(array_df_state[:limit_index + 1, col] == "correct")
        incorrect_index = np.where(array_df_state[:limit_index + 1, col] == "incorrect")
        process_index = np.where(array_df_state[:limit_index + 1, col] == "process")
        correct = len(correct_index[0])
        incorrect = len(incorrect_index[0])
        process = len(process_index[0])
        correct_num += correct
        incorrect_num += incorrect
        process_num += process
        if len(correct_index[0]) == 0 and len(incorrect_index[0]) == 0:
            non_ans += 1
        else:
            correct_ratio += correct / (incorrect + correct)
    ave_ans = (correct_num + incorrect_num) / n_cols
    correct_num = correct_num / n_cols
    incorrect_num = incorrect_num / n_cols
    correct_ratio = correct_ratio / n_cols

    return ave_ans, correct_num, incorrect_num, non_ans, correct_ratio, n_cols

def get_litesearch_data(df_node_state, df_prm_score, df_node_total_cost, max_cost):
    array_df_node_state = df_node_state.copy().values
    array_df_prm_score = df_prm_score.copy().values
    array_df_node_total_cost = df_node_total_cost.copy().values
    n_rows, n_cols = array_df_node_state.shape
    max_cost_len = int(max_cost)
    node_state_data = np.zeros((max_cost_len, n_cols))
    node_cost_data = np.zeros((max_cost_len, n_cols))
    for col in range(n_cols):
        answered_idx = np.where(array_df_node_state[:, col] != "process")[0]
        correct_idx = np.where(array_df_node_state[:, col] == "correct")[0]
        if len(answered_idx) == 0:
            continue
        info = {"cost": float(0), "target_score": float(0), "state": "process"}
        for i in range(len(answered_idx)):
            cost = array_df_node_total_cost[answered_idx[i], col]
            if cost > max_cost:
                break
            target_score = array_df_prm_score[answered_idx[i], col]
            if target_score >= 0.9:
                info = {
                    "cost": cost,
                    "target_score": target_score,
                    "state": array_df_node_state[answered_idx[i], col]
                }
                if info["state"] == "correct":
                    node_state_data[int(info["cost"]): , col] = 1
                    node_cost_data[int(info["cost"]), col] = 1
                    node_cost_data[int(info["cost"]) + 1: , col] = np.nan
                elif info["state"] == "incorrect":
                    node_state_data[int(info["cost"]): , col] = 0
                    node_cost_data[int(info["cost"]), col] = 0
                    node_cost_data[int(info["cost"]) + 1: , col] = np.nan

                break
            if target_score > info["target_score"]:
                info = {
                    "cost": cost,
                    "target_score": target_score,
                    "state": array_df_node_state[answered_idx[i], col]
                }
                if info["state"] == "correct":
                    node_state_data[int(info["cost"]): , col] = 1
                elif info["state"] == "incorrect":
                    node_state_data[int(info["cost"]): , col] = 0
    return node_state_data, node_cost_data

def main(
    max_cost: float = 10000,
    pic_name: str = "Results",
    path_to_graph: str = "./outputs/Math500",
    ):
    top_k = 1
    save_path: str =  "{exp_name}/data"
    if not Path(path_to_graph).exists():
        Path(path_to_graph).mkdir(parents=True, exist_ok=True)

    exp_name_list = os.environ["exp_paths_str"].split(",")
    graph_name_list = os.environ["graph_names_str"].split(",")

    cost_base_list = []
    score_base_list = []
    last_score_base_list = []
    num_nodes_list = []
    cost_data_lite_search = {}

    for i in range(len(exp_name_list)):
        exp_name = exp_name_list[i]
        name = graph_name_list[i]
        df_node_index = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_index.csv")
        )
        df_node_state = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_state.csv")
        )
        df_prm_score = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_prm_score.csv")
        )
        df_prm_score_min = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_prm_score_min.csv")
        )
        df_prm_score_total = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_prm_score_total.csv")
        )
        df_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_cost.csv")
        )
        df_total_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_total_cost.csv")
        )
        df_node_output_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_output_cost.csv")
        )
        df_node_total_output_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_total_output_cost.csv")
        )
        df_node_input_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_input_cost.csv")
        )
        df_node_total_input_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_total_input_cost.csv")
        )
        df_node_prm_output_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_prm_output_cost.csv")
        )
        df_node_prm_total_output_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_prm_total_output_cost.csv")
        )
        df_node_prm_input_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_prm_input_cost.csv")
        )
        df_node_prm_total_input_cost = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_prm_total_input_cost.csv")
        )
        df_node_depth = pd.read_csv(
            os.path.join(save_path.format(exp_name=exp_name), "df_node_depth.csv")
        )

        if "LiteSearch" in exp_name:
            last_socre_base_data, cost_data = get_litesearch_data(df_node_state, df_prm_score, df_node_total_output_cost, max_cost)
            cost_base_data = get_cost_base_data(df_node_state, df_node_total_output_cost, max_cost)
            cost_base_list.append([name, cost_base_data])
            last_score_base_list.append([name, last_socre_base_data])
            cost_data_lite_search[name] = cost_data
        else:
            last_socre_base_data = get_score_base_data(df_node_state, df_prm_score, df_node_total_output_cost, max_cost)
            cost_base_data = get_cost_base_data(df_node_state, df_node_total_output_cost, max_cost)

            cost_base_list.append([name, cost_base_data])
            last_score_base_list.append([name, last_socre_base_data])

        num_nodes_list.append(average_num_node(df_total_cost, max_cost))
        ans_num, correct_num, incorrect_num, non_ans_num, correct_ratio, num_prob = state_num_node(df_node_state, df_total_cost, max_cost)

    # ==========Graph==========
    title = ""
    save_path =f"{path_to_graph}/{pic_name}_result.pdf"
    plt.figure(figsize=(10, 12)) 
    x_values = np.arange(0, (last_score_base_list[0][1].shape[0] + 1), 1)
    for i, item in enumerate(last_score_base_list):
        x_values = np.arange(0, (last_score_base_list[i][1].shape[0] + 1), 1)
        if "LiteSearch" in exp_name_list[i]:
            total_cost = 0
            last_score_data = item[1]
            cost_data = cost_data_lite_search[item[0]]
            nan_count_per_row = np.sum(np.isnan(cost_data), axis=1)
            total_cost = np.sum(~np.isnan(cost_data)) / num_prob
            y_values = item[1].mean(axis=1)  
            y_values = np.insert(y_values, 0, 0)
            x_values = np.zeros((last_score_base_list[0][1].shape[0] + 1))
            for j in range(last_score_base_list[0][1].shape[0]):
                x_values[j+1] = x_values[j] + (num_prob - nan_count_per_row[j]) / num_prob
            plt.plot(x_values, y_values, linestyle='-', linewidth=1, label= item[0])
            title += f"{item[0]}: {y_values[-1]:.3f}, average cost: {total_cost}\n"
        else:
            y_values = item[1].mean(axis=1)  
            y_values = np.insert(y_values, 0, 0)
            plt.plot(x_values, y_values, linestyle='-', linewidth=1, label= item[0])
            title += f"{item[0]}: {y_values[-1]:.4f}\n"
    

    plt.xlabel("Total Output Cost")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.legend(loc='upper left') 
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    Fire(main)

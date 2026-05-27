import datetime
import json
import logging
import pickle
import sys
import time
from functools import partial
from pathlib import Path
import copy
from math import sqrt
import os
import hydra

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

folder_path = os.environ.get("FOLDER_PATH")
if folder_path:
    repo_root = Path(folder_path).expanduser().resolve()
else:
    repo_root = REPO_ROOT
    print("Error: FOLDER_PATH is not defined. Falling back to the repository root.")

for extra_path in (repo_root / "src", repo_root / "treesearch" / "src"):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.append(extra_path_str)

import treesearch as tq

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bg_mcts.llm.vllm_builder import build_model
from bg_mcts.llm_generation_interface import GenerationRequest, GenerationResult
from bg_mcts.prompts.base import PromptTemplate
from bg_mcts.prompts.prompt_configs import PromptConfig
from bg_mcts.tasks.math500.task import Math500Problem
from bg_mcts.tasks.aime24_25.task import AIMEProblem
from bg_mcts.tasks.minervamath.task import MinervamathProblem

sys.path.append(str(Path(__file__)))
from prompt import BaselinePrompt
from utils import NodeState, is_power_of_two


sys.setrecursionlimit(
    20000
)  # Example: Increase limit to 20000.  Choose a sensible value.


logger = logging.getLogger(__name__)

# Global variables to track total cost
total_cost = 0.0
total_output_cost = 0.0
total_input_cost = 0.0
prm_total_output_cost = 0.0
prm_total_input_cost = 0.0
cost_by_model: dict[str, float] = {}

# Global variables to track execution time
total_time = 0.0
time_by_model: dict[str, float] = {}
node_times: list[float] = []


def generate_fn(
    parent_node,
    task: Math500Problem,
    prompt_template: PromptTemplate,
    model_name: str,
    llm_log_dir: Path,
    stopping_rule: str,
    limit_info: dict,
    model
) -> tuple[NodeState, float]:
    global total_cost, total_output_cost, total_input_cost, prm_total_output_cost, prm_total_input_cost, cost_by_model, time_by_model, node_times

    start_time = time.time()

    # From root
    cost = float(0)
    if parent_node.state is None:
        new_messages = [
            {"role": "user", "content": prompt_template.initial_prompt()},
            {"role": "assistant", "content": "Step 1:"}
        ]
        cost = 4
        total_cost += cost
    else:
        pro_or_answer = parent_node.state.generation_result.generation_state
        new_messages = copy.deepcopy(parent_node.state.generation_result.request.messages)
        # if the parent node contains a final answer
        if pro_or_answer[0] == "correct" or pro_or_answer[0] == "incorrect":
            new_messages[-1]["content"] += parent_node.state.generation_result.generation + prompt_template.continue_prompt()
            # add continuing cost of prompt
            if model.model_name in {"google/gemma-3-4b-it", "google/gemma-3-12b-it"}:
                cost = 16
                total_cost += cost
            else:
                cost = 15
                total_cost += cost
        else:
            new_messages[-1]["content"] += parent_node.state.generation_result.generation
    
    if limit_info["limit_rule"] == "Cost_Base":
        if float(limit_info["limit_setting"] > total_cost):
            left_cost = float(limit_info["limit_setting"] - total_cost)
        else:
            cost = cost - (total_cost - limit_info["limit_setting"])
            total_cost = float(limit_info["limit_setting"])
            result = GenerationResult(
                request=GenerationRequest(
                    messages=new_messages, 
                    evaluation_messages=parent_node.state.generation_result.request.evaluation_messages if parent_node.state is not None else []
                    ), 
                generation="", generation_state=["process",""], 
            )
            eval_result = float(0)
            score = float(0)
            scores = [float(0)]
            output_cost = float(0)
            total_output_cost = float(0)
            input_cost = float(0)
            total_input_cost = float(0)
            prm_output_cost = float(0)
            prm_total_output_cost = float(0)
            prm_input_cost = float(0)
            prm_total_input_cost = float(0)
            return NodeState(
                generation_result=result, eval_results=eval_result, model_name=model_name
            ), score, scores, cost, total_cost, output_cost, total_output_cost, input_cost, total_input_cost, prm_output_cost, prm_total_output_cost, prm_input_cost, prm_total_input_cost
    else:
        left_cost = float(model.model_setting["max_tokens"])

    if stopping_rule == "full":
        generation, output_cost, input_cost, generation_state = model.generate(messages=new_messages, stopping=False, left_cost=left_cost)
    elif stopping_rule == "sequential":
        generation, output_cost, input_cost, generation_state = model.generate(messages=new_messages, stopping=True, left_cost=left_cost)


    if model_name not in cost_by_model:
        cost_by_model[model.model_name] = 0.0
    cost_by_model[model.model_name] += output_cost

    result = GenerationResult(
        request=GenerationRequest(
            messages=new_messages, 
            evaluation_messages=parent_node.state.generation_result.request.evaluation_messages if parent_node.state is not None else []
            ), 
        generation=generation, generation_state=generation_state, 
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
        :-3
    ]  # up to milliseconds

    log_txt = llm_log_dir / f"log_{timestamp}_{model_name}.txt"
    log_txt.parent.mkdir(parents=True, exist_ok=True)
    log_txt.write_text(
        json.dumps(
            {"model": model_name, "output_cost": output_cost, "result": generation},
            indent=4,
        )
    )  # save cost and result

    # evaluate the results of generation
    eval_result, eval_results, generation_state, evaluation_messages, prm_output_cost, prm_input_cost = task.generate_eval_results(llm_answer=result, kind="transform", generation_state=generation_state, parent_node=parent_node, stopping_rule=stopping_rule)

    score = eval_result
    scores = eval_results
    result.generation_state = generation_state
    result.request.evaluation_messages = evaluation_messages

    # Update cost info
    total_output_cost += output_cost
    total_input_cost += input_cost
    prm_total_output_cost += prm_output_cost
    prm_total_input_cost += prm_input_cost
    
    cost = output_cost
    total_cost += output_cost

    # Calculate execution time for this node
    execution_time = time.time() - start_time
    node_times.append(execution_time)

    # Update time by model
    if model_name not in time_by_model:
        time_by_model[model.model_name] = 0.0
    time_by_model[model.model_name] += execution_time

    return NodeState(
        generation_result=result, eval_results=eval_result, model_name=model_name
    ), score, scores, cost, total_cost, output_cost, total_output_cost, input_cost, total_input_cost, prm_output_cost, prm_total_output_cost, prm_input_cost, prm_total_input_cost


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    global total_cost, total_output_cost, total_input_cost, prm_total_output_cost, prm_total_input_cost, cost_by_model, total_time, time_by_model, node_times

    # Reset global cost and time trackers
    total_cost = 0.0
    total_output_cost = 0.0
    total_input_cost = 0.0
    prm_total_output_cost = 0.0
    prm_total_input_cost = 0.0
    cost_by_model = {}
    total_time = 0.0
    time_by_model = {}
    node_times = []

    start_time = time.time()

    # Models
    models_config = cfg["models"]
    if isinstance(models_config, dict):
        models_config = [models_config]

    model_setting = OmegaConf.to_container(cfg["vllm_setting"], resolve=True)

    # PRM
    prm_name = str(cfg["prms"][0]["name"])
    gen_prm_setting = OmegaConf.to_container(cfg["gen_prm_setting"], resolve=True)

    # Task
    task_setting = cfg["task_setting"]
    task_name = str(task_setting["task_name"])
    task_id = str(task_setting["task_id"])
    
    if task_id == "inf":
        # task_id: 13e47133 is recognized as inf
        task_id = os.getenv("TASK_ID")
    
    if task_name == "Math500":
        math500_problem_path = repo_root / "benchmarks" / "Math500" / "info" / f"{task_id}.json"
        if not math500_problem_path.exists():
            print(f"Task {task_id} not found")
            sys.exit(1)
        task = Math500Problem.load_file(json_path=math500_problem_path, prm_name=prm_name, prm_setting=gen_prm_setting)
    elif task_name == "AIME24_25":
        aime24_25_problem_path = repo_root / "benchmarks" / "AIME24_25" / "info" / f"{task_id}.json"
        if not aime24_25_problem_path.exists():
            print(f"Task {task_id} not found")
            sys.exit(1)
        task = AIMEProblem.load_file(json_path=aime24_25_problem_path, prm_name=prm_name, prm_setting=gen_prm_setting)
    elif task_name == "Minervamath":
        minervamath_problem_path = repo_root / "benchmarks" / "Minervamath" / "info" / f"{task_id}.json"
        if not minervamath_problem_path.exists():
            print(f"Task {task_id} not found")
            sys.exit(1)
        task = MinervamathProblem.load_file(json_path=minervamath_problem_path, prm_name=prm_name, prm_setting=gen_prm_setting)


    # prompt
    prompt_template = BaselinePrompt(prompt_config=PromptConfig(), data=task)

    # Algo
    algo_config = cfg["algo"]
    algo_cls = getattr(tq, algo_config["class_name"])
    if algo_config["class_name"] == "StandardMCTS":
        samples_per_action = int(algo_config["params"]["samples_per_action"])
        exploration_weight = sqrt(float(algo_config["params"]["exploration_weight"]))
        algo = algo_cls(samples_per_action=samples_per_action, exploration_weight=exploration_weight)
    elif algo_config["class_name"] == "BGMCTS":
        samples_per_action = int(algo_config["params"]["samples_per_action"])
        exploration_weight = int(algo_config["params"]["exploration_weight"])
        depth_weight = float(algo_config["params"]["depth_weight"])
        variance_weight = float(algo_config["params"]["variance_weight"])
        budget = float(algo_config["params"]["budget"])
        algo = algo_cls(
            samples_per_action=samples_per_action,
            exploration_weight=exploration_weight,
            depth_weight=depth_weight,
            variance_weight=variance_weight,
            budget=budget
        )
    elif algo_config["class_name"] == "TreeOfThoughtsBFSAlgo":
        breadth_limit = int(algo_config["params"]["breadth_limit"])
        size_limit = int(algo_config["params"]["size_limit"])
        algo = algo_cls(breadth_limit=breadth_limit, size_limit=size_limit)
    elif "LiteSearch" in algo_config["class_name"]:
        max_expansion = int(algo_config["params"]["max_expansion"])
        lmbd = float(algo_config["params"]["lambda"])
        eps = float(algo_config["params"]["epsilon"])
        algo = algo_cls(budget=max_expansion, lmbd=lmbd, eps=eps)
    else:
        algo = algo_cls()

    # stopping rule setting
    stopping_rule = str(cfg["limit_setting"]["stopping_rule"])

    # set the place of Hydra log
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    for subdir in ["llm_logs", "costs", "checkpoints"]:
        if not (save_dir / subdir).exists():
            (save_dir / subdir).mkdir()

    llm_log_dir = save_dir / "llm_logs"

    models = {}
    for model_config in models_config:
        models[model_config["name"]] = build_model(model_name=model_config["name"], model_setting=model_setting, environment=str(cfg["environment"]))

    # search tree iniialization and the procedure of researchs
    if not cfg["checkpoint_path"]: 
        search_tree = algo.init_tree()
        print("Initialized state")
    else:
        with open(cfg["checkpoint_path"], "rb") as f:
            search_tree = pickle.load(f)
        print(f"Loaded checkpoint from {cfg['checkpoint_path']}")
        # get cost so far
        if (save_dir / "cost_summary.json").exists():
            with open(save_dir/"cost_summary.json", "r") as f:
                cost_summary = json.load(f)
                total_cost = cost_summary["total_cost"]
                total_output_cost = cost_summary["total_output_cost"]
                total_input_cost = cost_summary["total_input_cost"]
                prm_total_output_cost = cost_summary["prm_total_output_cost"]
                prm_total_input_cost = cost_summary["prm_total_input_cost"]
                cost_by_model = cost_summary["cost_by_model"]
                
        # get time so far if available
        time_summary_path = save_dir /"costs"/ "time_summary.json"
        if time_summary_path.exists():
            with open(time_summary_path, "r") as f:
                time_summary = json.load(f)
                total_time = time_summary.get("total_time", 0.0)
                time_by_model = time_summary.get("time_by_model", {})
                node_times = time_summary.get("node_times", [])

    limit_rule = cfg["limit_setting"]["limit_rule"]
    if limit_rule == "Node_Base":
        max_num_nodes = cfg["limit_setting"]["max_num_nodes"]
        initial_num_nodes = len(algo.get_state_score_pairs(search_tree))
        limit_setting = (max_num_nodes - initial_num_nodes)
        limit_info = {"limit_rule": "Node_Base", "limit_setting": max_num_nodes}
        progress_state = 0
    elif limit_rule == "Cost_Base":
        max_num_cost = cfg["limit_setting"]["max_num_cost"]
        limit_setting = max_num_cost
        progress_state = total_cost
        limit_info = {"limit_rule": "Cost_Base", "limit_setting": max_num_cost}

    generate_fns = {
        model_config["name"]: partial(
            generate_fn,
            task=task,
            model_name=model_config["name"],
            llm_log_dir=llm_log_dir,
            prompt_template=prompt_template,
            stopping_rule=stopping_rule,
            limit_info=limit_info,
            model = models[model_config["name"]]
        )
        for model_config in models_config
    }

    pbar = tqdm(total=limit_setting)
    while progress_state < limit_setting:
        node_start_time = time.time()
        search_tree = algo.step(search_tree, generate_fns)
        n_answers = len(algo.get_state_score_pairs(search_tree))

        node_execution_time = time.time() - node_start_time
        node_times.append(node_execution_time)

        print_log = False
        if limit_rule == "Node_Base":
            progress_state += 1
            if n_answers == max_num_nodes:
                print_log = True
        elif limit_rule == "Cost_Base":
            progress_state = total_cost
            if total_cost >= max_num_cost:
                print_log = True
        

        if n_answers % 10 == 0 or is_power_of_two(n_answers) or print_log:
            with open(
                save_dir / "checkpoints" / f"checkpoint_n_answers_{n_answers}.pkl", "wb"
            ) as f:
                pickle.dump(search_tree, f)
            with open(save_dir / "checkpoints" / f"checkpoint_latest.pkl", "wb") as f:
                pickle.dump(search_tree, f)
                # Update total time
            if print_log:
                with open(
                    save_dir / "checkpoints" / f"checkpoint_total.pkl", "wb"
                ) as f:
                    pickle.dump(search_tree, f)
                with open(
                    save_dir / f"checkpoint_total.pkl", "wb"
                ) as f:
                    pickle.dump(search_tree, f)
            
            total_time = time.time() - start_time

            # Log accumulated cost and time every 10 steps
            logger.info(f"Current total cost: ${total_cost:.6f}")
            logger.info(f"Current total output cost: ${total_output_cost:.6f}")
            logger.info(f"Current total intput cost: ${total_input_cost:.6f}")
            logger.info(f"Current PRM total output cost: ${prm_total_output_cost:.6f}")
            logger.info(f"Current PRM total input cost: ${prm_total_input_cost:.6f}")
            logger.info(f"Current total time: {total_time:.2f} seconds")
            for model, model_cost in cost_by_model.items():
                logger.info(f"  {model} cost: ${model_cost:.6f}")
            for model, model_time in time_by_model.items():
                logger.info(f"  {model} time: {model_time:.2f} seconds")

            # Save cost summary to a JSON file
            cost_summary = {
                "total_cost": total_cost, 
                "total_output_cost": total_output_cost, 
                "total_input_cost": total_input_cost, 
                "prm_total_output_cost": prm_total_output_cost, 
                "prm_total_input_cost": prm_total_input_cost, 
                "cost_by_model": cost_by_model,
                "total_node": len(algo.get_state_score_pairs(search_tree))
            }
            with open(
                save_dir / "costs" / f"cost_summary_n_answers_{n_answers}.json", "w"
            ) as f:
                json.dump(cost_summary, f, indent=2)
            with open(save_dir / f"cost_summary.json", "w") as f:
                json.dump(cost_summary, f, indent=2)

            # Save time summary to a JSON file
            time_summary = {
                "total_time": total_time,
                "total_time_minutes": total_time / 60,
                "total_time_hours": total_time / 3600,
                "time_by_model": time_by_model,
                "time_by_model_minutes": {
                    model: time / 60 for model, time in time_by_model.items()
                },
                "node_times": node_times,
                "avg_node_time": sum(node_times) / len(node_times) if node_times else 0,
                "avg_node_time_minutes": (
                    sum(node_times) / len(node_times) if node_times else 0
                )
                / 60,
            }
            with open(
                save_dir / "costs" / f"time_summary_n_answers_{n_answers}.json", "w"
            ) as f:
                json.dump(time_summary, f, indent=2)
            with open(save_dir / "costs" / f"time_summary.json", "w") as f:
                json.dump(time_summary, f, indent=2)

        pbar.n = progress_state          
        pbar.refresh()

    pbar.close()

    # Update final total time
    total_time = time.time() - start_time

    # Log the final total cost and time
    logger.info("===== Final Cost Summary =====")
    logger.info(f"Total LLM cost: ${total_cost:.6f}")
    for model, model_cost in cost_by_model.items():
        logger.info(f"  {model}: ${model_cost:.6f}")

    logger.info("===== Final Time Summary =====")
    logger.info(
        f"Total execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes, {total_time / 3600:.2f} hours)"
    )
    logger.info(
        f"Average node time: {sum(node_times) / len(node_times) if node_times else 0:.2f} seconds ({(sum(node_times) / len(node_times) if node_times else 0) / 60:.2f} minutes)"
    )
    for model, model_time in time_by_model.items():
        logger.info(
            f"  {model}: {model_time:.2f} seconds ({model_time / 60:.2f} minutes)"
        )

    # Save cost summary to a JSON file
    cost_summary = {
        "total_cost": total_cost, 
        "total_output_cost": total_output_cost, 
        "total_input_cost": total_input_cost, 
        "prm_total_output_cost": prm_total_output_cost, 
        "prm_total_input_cost": prm_total_input_cost, 
        "cost_by_model": cost_by_model,
        "total_node": len(algo.get_state_score_pairs(search_tree))
    }
    with open(save_dir / "cost_summary.json", "w") as f:
        json.dump(cost_summary, f, indent=2)

    # Save time summary to a JSON file
    time_summary = {
        "total_time": total_time,
        "total_time_minutes": total_time / 60,
        "total_time_hours": total_time / 3600,
        "time_by_model": time_by_model,
        "time_by_model_minutes": {
            model: time / 60 for model, time in time_by_model.items()
        },
        "node_times": node_times,
        "avg_node_time": sum(node_times) / len(node_times) if node_times else 0,
        "avg_node_time_minutes": (
            sum(node_times) / len(node_times) if node_times else 0
        )
        / 60,
    }
    with open(save_dir / "time_summary.json", "w") as f:
        json.dump(time_summary, f, indent=2)

    if str(cfg["visualize"]) == "True":
        for extra_path in (repo_root / "src", repo_root / "treesearch" / "src"):
            extra_path_str = str(extra_path)
            if extra_path_str not in sys.path:
                sys.path.append(extra_path_str)

        from treesearch import visualization
        pic_path = Path(str(cfg["tree_photo_path"]))
        pic_path.mkdir(parents=True, exist_ok=True)
        pic_path = pic_path /  task_id

        visualization.visualize_tree_paper(
            tree=search_tree,
            save_path=pic_path,
            format="pdf",
            nodesep=0,
            ranksep=-1,
            edge="none",
            budget=max_num_cost,
            number=True,
            correct_color="black",
            incorrect_color="black",
            correct_shape = "star",
            incorrect_shape = "triangle",
            penwidth="1",
            stopping=False,
        )
        print("Visualized📝")
if __name__ == "__main__":
    main()

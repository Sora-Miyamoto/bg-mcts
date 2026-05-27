from pathlib import Path
from fire import Fire


def make_ckpt_path(exp_path, task_id):
    return f"{exp_path}/results/{task_id}/checkpoints/checkpoint_total.pkl"


def main(exp_path, task_file):
    exp_path_obj = Path(exp_path).expanduser().resolve()
    task_file_obj = Path(task_file).expanduser().resolve()

    if not exp_path_obj.exists():
        raise FileNotFoundError(f"experiment path does not exist: {exp_path_obj}")
    if not exp_path_obj.is_dir():
        raise NotADirectoryError(f"experiment path is not a directory: {exp_path_obj}")
    if not task_file_obj.exists():
        raise FileNotFoundError(f"task file does not exist: {task_file_obj}")
    if not task_file_obj.is_file():
        raise FileNotFoundError(f"task file is not a file: {task_file_obj}")

    with task_file_obj.open("r") as f:
        task_list = [t.strip() for t in f if t.strip()]

    if not task_list:
        raise ValueError(f"task file is empty: {task_file_obj}")

    unfinished_tasks = []
    for task_id in task_list:
        path_to_ckpt = make_ckpt_path(exp_path_obj, task_id)
        state_path = Path(path_to_ckpt)
        if not state_path.exists():
            unfinished_tasks.append(task_id)

    completed_tasks = len(task_list) - len(unfinished_tasks)
    status = "finished" if not unfinished_tasks else "in progress"

    print("=== Experiment Progress Check ===")
    print(f"experiment path: {exp_path_obj}")
    print(f"task file: {task_file_obj}")
    print(f"total tasks: {len(task_list)}")
    print(f"completed tasks: {completed_tasks}")
    print(f"unfinished tasks count: {len(unfinished_tasks)}")
    print(f"status: {status}")
    print("unfinished task list:")
    if unfinished_tasks:
        for task_id in unfinished_tasks:
            print(task_id)
    else:
        print("none")

if __name__ == "__main__":
    Fire(main)

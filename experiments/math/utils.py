import math
from dataclasses import dataclass

import treesearch as tq

from bg_mcts.eval_result import EvalResult


def is_power_of_two(n: int):
    return n > 0 and (n & (n - 1)) == 0


def get_top_k(state, algorithm, k):
    if hasattr(state, "tree"):
        # nodes in ascending order
        nodes = state.tree.get_nodes()
        nodes.sort(key=lambda x: x.score, reverse=True)
        top_k = [(node.state, node.score) for node in nodes[:k]]
        return top_k
    else:
        return tq.top_k(state, algorithm, k=k)


@dataclass
class NodeState:
    generation_result: str
    eval_results: EvalResult
    model_name: str

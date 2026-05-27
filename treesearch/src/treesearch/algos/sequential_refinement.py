import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, log, sqrt
from random import shuffle
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

from treesearch.algos.base import Algorithm
from treesearch.algos.tree import Node, Tree
from treesearch.types import GenerateFnType, StateScoreType, StateScoreCostType

# Type variable for state
StateT = TypeVar("StateT")


def softmax(values: List[float]) -> List[float]:
    """
    Compute softmax values for a list of scores.

    Args:
        values: List of scores

    Returns:
        List of softmax probabilities
    """
    # Shift values for numerical stability (prevent overflow)
    shifted = [x - max(values) for x in values]
    exp_values = [exp(x) for x in shifted]
    sum_exp = sum(exp_values)
    return [x / sum_exp for x in exp_values]


@dataclass
class SequentialRefinement_state(Generic[StateT]):
    """State for Monte Carlo Tree Search algorithm."""

    tree: Tree[StateT]
    
    priors: Dict[int, float] = field(default_factory=dict)
    next_nodes: List[Node[StateT]] = field(default_factory=list)
    # next_nodes: List[Tuple[Node[StateT], str]] = dataclasses.field(default_factory=list)

class SequentialRefinement(Algorithm[StateT, SequentialRefinement_state[StateT]]):
    """
    Standard Monte Carlo Tree Search (MCTS) algorithm with UCT scoring.

    This implementation uses the Upper Confidence Bound for Trees (UCT)
    formula to balance exploration and exploitation.
    """

    def __init__(self, *, exploration_weight: float = sqrt(2)):

        self.exploration_weight = exploration_weight

    def step(
        self,
        state: SequentialRefinement_state,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> SequentialRefinement_state:
        """
        Generate one additional node and add that to a given state.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        if not state.next_nodes:
            node = state.tree.root
        else:
            node = state.next_nodes[-1]

        # Select the next action using UCB
        action = list(generate_fn.keys())[0]


        # Generate a new state and add it to the tree
        new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost = generate_fn[action](node)
        new_node = state.tree.add_node(
            (new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost),
            node
        )
        
        # Update scores for the selected action
        state.next_nodes.append(new_node)

        return state


    def init_tree(self) -> SequentialRefinement_state:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return SequentialRefinement_state(tree)

    def get_state_score_pairs(self, state: SequentialRefinement_state) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()

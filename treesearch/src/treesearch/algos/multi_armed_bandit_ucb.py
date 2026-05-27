import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from math import log, sqrt
from typing import Dict, Generic, List, Tuple, TypeVar

from treesearch.algos.base import Algorithm
from treesearch.algos.tree import Node, Tree
from treesearch.types import GenerateFnType, StateScoreType, StateScoreCostType

# Type variable for state
StateT = TypeVar("StateT")


@dataclass
class UCBState(Generic[StateT]):
    """State for Multi-Armed Bandit UCB algorithm."""

    tree: Tree[StateT]
    scores_by_action: Dict[str, List[float]] = dataclasses.field(default_factory=dict)
    next_nodes: List[Tuple[Node[StateT], str]] = dataclasses.field(default_factory=list)


class MultiArmedBanditUCBAlgo(Algorithm[StateT, UCBState[StateT]]):
    """
    Multi-Armed Bandit algorithm with Upper Confidence Bound (UCB) implementation.

    Balances exploration and exploitation using the UCB formula.
    """

    def __init__(self, *, exploration_weight: float = sqrt(2)):
        """
        Initialize the Multi-Armed Bandit UCB algorithm.

        Args:
            exploration_weight: Weight for the exploration term in UCB formula
        """
        self.exploration_weight = exploration_weight

    def step(
        self,
        state: UCBState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> UCBState:
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

        # Initialize scores for actions if not already done
        for action in generate_fn:
            if action not in state.scores_by_action:
                state.scores_by_action[action] = []

        # Select the next action using UCB
        action = self._select_action(state, list(generate_fn.keys()))

        # Use the root node if the tree is empty
        if not state.tree.root.children:
            parent = state.tree.root
        else:
            # In this simple version, we always expand from the root
            parent = state.tree.root

        # Generate a new state and add it to the tree
        new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost = generate_fn[action](parent)
        state.tree.add_node(
            (new_state, new_score, new_scores, new_cost, new_total_cost, new_total_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost), 
            parent
        )
        
        # Update scores for the selected action
        state.scores_by_action[action].append(new_score)

        return state

    def _select_action(self, state: UCBState, actions: List[str]) -> str:
        """
        Select the next action using the UCB formula.

        Args:
            state: Current algorithm state
            actions: List of available actions

        Returns:
            Selected action
        """
        # If any action hasn't been tried yet, select it
        for action in actions:
            if len(state.scores_by_action[action]) == 0:
                return action

        # Calculate total number of trials
        total_trials = sum(len(scores) for scores in state.scores_by_action.values())

        # Find action with highest UCB score
        max_ucb_score = -float("inf")
        best_action = actions[0]  # Default to first action

        for action in actions:
            scores = state.scores_by_action[action]
            # Calculate average score (exploitation term)
            avg_score = sum(scores) / len(scores)
            # Calculate exploration term
            exploration = self.exploration_weight * sqrt(
                log(total_trials) / len(scores)
            )
            # UCB score combines exploitation and exploration
            ucb_score = avg_score + exploration

            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_action = action

        return best_action

    def init_tree(self) -> UCBState:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return UCBState(tree)

    def get_state_score_pairs(self, state: UCBState) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()

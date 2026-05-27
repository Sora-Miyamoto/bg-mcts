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

@dataclass
class LiteSearchState(Generic[StateT]):
    """State for LiteSearch algorithm."""

    tree: Tree[StateT]
    value_state: Dict[int, float] = field(default_factory=dict)
    limit_state: Dict[int, float] = field(default_factory=dict)
    leaf_node: list[int] = field(default_factory=list) 
    next_nodes: List[Tuple[Node[StateT], str]] = field(default_factory=list)

class LiteSearchBatch(Algorithm[StateT, LiteSearchState[StateT]]):
    """
    LiteSearch Batch algorithm
    Original paper: https://arxiv.org/abs/2407.00320
    """

    def __init__(
        self, 
        *,
        budget: int = 10,
        lmbd: float = 0,
        eps: float = 0.9,
    ):
        """
        Initialize the Lite Search algorithm.

        Args:
            budget: the limit of nodes each leaf node can expand
            lmbd: highper parameter of step weight
            eps: highper parameter to stop the expansion of the node
        """
        self.budget = budget
        self.lmbd = lmbd
        self.eps = eps
        print(f"Initialized LiteSearch Batch with budget={self.budget}, lambda={self.lmbd}, epsilon={self.eps}")

    def step(
        self,
        state: LiteSearchState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> LiteSearchState:
        """
        Perform one step of the Lite Search algorithm.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        if not state.next_nodes:
            action = list(generate_fn.keys())[0]
            node = self._select(state)
            state.next_nodes.append((node,action))
            
        node, action = state.next_nodes.pop(0) 
        # Simulation: Generate a new state using the selected action
        new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost = generate_fn[action](node)

        # Add the new node to the tree
        new_node = state.tree.add_node(
            (new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost),
            node
        )

        # Update statistics for the new node
        node_id = new_node.expand_idx
        value = self._calc_value(new_node)
        state.value_state[node_id] = value
        state.limit_state[node_id] = self._calc_limit(state, new_node)
        state.leaf_node.append(node_id)

        parent = new_node.parent
        parent_id = parent.expand_idx

        if len(parent.children) >= state.limit_state[parent_id]:
            state.leaf_node.remove(parent_id)
        else:
            action = list(generate_fn.keys())[0]
            state.next_nodes.append((parent,action))

        return state

    def _select(self, state: LiteSearchState) -> Node:
        """
        Select a node to expand
        
        Args:
            state: Current algorithm state

        Returns:
            Selected node
        """
        if len(state.leaf_node) == 0:
            node = state.tree.root
            node_id = node.expand_idx
            state.leaf_node.append(node_id)
            value = self._calc_value(node)
            state.value_state[node_id] = 0
            state.limit_state[node_id] = self._calc_limit(state, node)
            return node
        
        selected_idx = max(
            (k for k in state.leaf_node if k in state.value_state),
            key=lambda k: state.value_state[k]
        )
        node = state.tree.get_node(selected_idx)
        return node

    def _calc_value(self, node: Node):
        depth = node.depth
        if node.expand_idx == -1:
            return 0
        value = node.score
        return value

    def _calc_limit(self, state: LiteSearchState, node: Node):
        depth = node.depth
        if node.expand_idx == -1:
            return self.budget
        dynamic_limit = int(log(1 - self.eps) / (depth * log(1 - self._calc_value(node))))
        if dynamic_limit <= self.budget:
            if dynamic_limit < 1:
                return 1
            else:
                return dynamic_limit
        else:
            return self.budget
        
    def init_tree(self) -> LiteSearchState:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return LiteSearchState(tree=tree)

    def get_state_score_pairs(self, state: LiteSearchState) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()

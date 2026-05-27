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
class BGMCTS_state(Generic[StateT]):
    """State for Budget Guided Monte Carlo Tree Search algorithm."""

    tree: Tree[StateT]
    visit_counts: Dict[int, int] = field(default_factory=dict)
    value_sums: Dict[int, float] = field(default_factory=dict)
    priors: Dict[int, float] = field(default_factory=dict)
    next_nodes: List[Tuple[Node[StateT], str]] = field(default_factory=list)
    subtree_node: Dict[int, list] = field(default_factory=dict)  
    depth_node: Dict[int, int] = field(default_factory=dict)
    cost_used: float = 0
    ans_depth_list: List[int] = field(default_factory=list)
    ans_depth: float = 0


class BGMCTS(Algorithm[StateT, BGMCTS_state[StateT]]):
    """
    Budget Guided Monte Carlo Tree Search (MCTS) algorithm
    Original paper: https://arxiv.org/abs/2602.09574
    """

    def __init__(
        self,
        *,
        samples_per_action: int = 2,
        exploration_weight: int = 2,
        depth_weight: float = 1.0,
        variance_weight: float = 1.0,
        budget: float = 30000,
    ):
        """
        Initialize the MCTS algorithm.

        Args:
            samples_per_action: Number of samples to generate for each action
            exploration_weight (c): Weight for the exploration term in UCT formula
            depth_weight (kappa): Weight for the depth term in BG-PUCT formula
            variance_weight (lambda): Weight for the variance term in gen node score formula
            budget: Given budget for the search, used to scale exploration and depth terms
        """
        self.samples_per_action = samples_per_action
        self.exploration_weight = sqrt(exploration_weight)
        self.depth_weight = depth_weight
        self.variance_weight = variance_weight
        self.budget = budget

        print(f"Initialized BGMCTS with samples_per_action={self.samples_per_action}, exploration_weight={self.exploration_weight}, depth_weight={self.depth_weight}, variance_weight={self.variance_weight}, budget={self.budget}")

    def step(
        self,
        state: BGMCTS_state,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> BGMCTS_state:
        """
        Perform one step of the MCTS algorithm.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        # If no nodes are queued for expansion, select nodes to expand
        if not state.next_nodes:
            # Selection: Find the most promising node to expand
            node = self._select(state)
            if len(node.children) == 0:
                expand_num = self.samples_per_action
            else:
                expand_num = 1
            # Create pairs of (node, action) for all actions
            pairs_to_add = []
            actions = list(generate_fn.keys())

            # For each sample, add all actions
            for _ in range(expand_num):
                for action in actions:
                    pairs_to_add.append((node, action))

            # Shuffle the pairs to add randomness to expansion order
            shuffle(pairs_to_add)
            state.next_nodes.extend(pairs_to_add)
                
        # Get the next node and action to expand
        node, action = state.next_nodes.pop(0)

        # Simulation: Generate a new state using the selected action
        new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost = generate_fn[action](node)

        # Add the new node to the tree
        new_node = state.tree.add_node(
            (new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost),
            node
        )

        # Update Cost term
        state.cost_used = new_total_cost

        # Update statistics for the new node
        node_id = new_node.expand_idx
        state.visit_counts[node_id] = 1
        state.value_sums[node_id] = new_score
        state.depth_node[node_id] = self._get_depth(new_node)
        state.subtree_node[node_id] = [node_id]

        generation_state = new_node.state.generation_result.generation_state
        if generation_state[0] != "process":
            parent_id = new_node.parent.expand_idx
            state.ans_depth_list.append(state.depth_node[parent_id] + 1)
            # state.ans_depth_list.append(new_node.depth)
            state.ans_depth = sum(state.ans_depth_list) / len(state.ans_depth_list)
        elif not state.ans_depth_list and new_node.depth > state.ans_depth:
            state.ans_depth = new_node.depth

        if not state.next_nodes:
            parent = new_node.parent
            target = new_node
            for item in parent.children[:-1]:
                current = item
                if current.score > target.score:
                    target = current
                elif current.score == target.score and current.cost < target.cost:
                    target = current
                
        # Backpropagation: Update statistics for all nodes in the path
        self._backpropagate(state, new_node, new_score)

        # Update priors if this node has siblings
        parent = new_node.parent
        if parent and len(parent.children) > 1:
            self._update_priors(state, parent)

        return state

    def _update_priors(self, state: BGMCTS_state, parent: Node) -> None:
        """
        Update prior probabilities for all children of a node using softmax.

        Args:
            state: Current algorithm state
            parent: Parent node whose children's priors will be updated
        """
        children = parent.children
        scores = [child.score for child in children]
        priors = softmax(scores)

        for child, prior in zip(children, priors):
            state.priors[child.expand_idx] = prior

    def _select(self, state: BGMCTS_state) -> Node:
        """
        Select a node to expand using Scores (BG-PUCT and E_{gen}).

        Starts from the root and selects child nodes with highest Scores
        until reaching a leaf node or generative node.

        Args:
            state: Current algorithm state

        Returns:
            Selected node
        """
        node = state.tree.root

        # If the tree is empty, return the root
        if not node.children:
            state.depth_node[node.expand_idx] = 0
            return node

        # Traverse down the tree selecting best child according to UCT
        while node.children:
            # We're selecting based on the UCT score, which balances exploration and exploitation.
            best_child = max(
                node.children, key=lambda child: self._bg_puct_score(state, child, node)
            )
            best_puct = self._bg_puct_score(state, best_child, node)
            gen_score = self._gen_node_score(state, node)
            if gen_score >= best_puct:
                break
            else:
                node = best_child

        return node

    def _bg_puct_score(self, state: BGMCTS_state, node: Node, parent: Node) -> float:
        """
        Calculate the UCT score for a node.
        """
        exploitation = 0
        for item in state.subtree_node[node.expand_idx]:
            current = state.tree.get_node(item)  
            exploitation += self._calc_depth_score(state, current)
        exploitation = exploitation / len(state.subtree_node[node.expand_idx])

        # Get visit counts
        parent_visits = state.visit_counts.get(parent.expand_idx, 1)
        node_visits = state.visit_counts.get(node.expand_idx, 1)

        # Get prior (default to 1.0 if not set)
        prior = state.priors.get(node.expand_idx, 1.0)

        # Calculate exploration term (weighted by prior)
        
        exploration = (
            self.exploration_weight * (1 - (state.cost_used / self.budget)) * prior * sqrt(log(parent_visits) / node_visits)
        )
        return exploitation + exploration
    
    def _gen_node_score(self, state: BGMCTS_state, parent: Node) -> float:
        children_scores = [item.score for item in parent.children]
        mean_value = sum(children_scores) / len(children_scores)
        variance = self._calc_variance(node=parent)
        variance_score = self.variance_weight * (1 - state.cost_used / self.budget) * variance
        return mean_value + variance_score

    def _calc_depth_score(self, state: BGMCTS_state, node: Node):
        # depth = node.depth
        depth = state.depth_node[node.expand_idx]
        return node.score + self.depth_weight * (state.cost_used / self.budget) * (depth / state.ans_depth)
        
    def _backpropagate(self, state: BGMCTS_state, node: Node, score: float) -> None:
        """
        Update statistics for all nodes in the path from node to root.

        Args:
            state: Current algorithm state
            node: Leaf node to start backpropagation from
            score: Score to backpropagate
        """
        current: Optional[Node] = node.parent
        
        while not current.is_root():
            node_id = current.expand_idx
            state.visit_counts[node_id] = state.visit_counts.get(node_id, 0) + 1
            state.value_sums[node_id] = state.value_sums.get(node_id, 0) + score
            state.subtree_node[node_id].append(node.expand_idx)
            current = current.parent

    def _get_depth(self, node: Node):
        current = node
        generation_state = current.state.generation_result.generation_state
        if generation_state[0] != "process":
            return 0
        depth = 0
        while generation_state[0] == "process" and current.expand_idx >=0:                             
            depth += 1
            current = current.parent
            if current.is_root():
                break
            generation_state = current.state.generation_result.generation_state
        return depth

    def _calc_variance(self, node: Node):
        """
        Calculate the variance of the scores of the children of a node.
        """
        children = node.children
        if len(children) == 0:
            variance = 0
        else:
            scores = []
            for item in children:
                scores.append(item.score)
            mean = sum(scores) / len(scores)
            variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance

    def init_tree(self) -> BGMCTS_state:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return BGMCTS_state(tree=tree)

    def get_state_score_pairs(self, state: BGMCTS_state) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()
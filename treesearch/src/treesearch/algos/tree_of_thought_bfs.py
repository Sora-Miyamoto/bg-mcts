import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from functools import total_ordering
from heapq import heappop, heappush
from typing import Generic, List, Tuple, TypeVar

from treesearch.algos.base import Algorithm
from treesearch.algos.tree import Node, Tree
from treesearch.types import GenerateFnType, StateScoreType, StateScoreCostType

# Type variable for state
StateT = TypeVar("StateT")


@total_ordering
class TreeOfThoughtsBFSHeapItem(Generic[StateT]):
    """
    Heap item for Tree of Thoughts BFS algorithm.

    Prioritizes:
    1. Deeper nodes (higher depth first)
    2. Higher scores when depths are equal
    """

    def __init__(self, node: Node[StateT]):
        self.node = node

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TreeOfThoughtsBFSHeapItem)
        return (
            self.node.depth == other.node.depth and self.node.score == other.node.score
        )

    def __lt__(
        self, other: "TreeOfThoughtsBFSHeapItem[StateT]"
    ) -> bool:  # heapq is a min heap
        if self.node.depth != other.node.depth:
            return (
                self.node.depth > other.node.depth
            )  # Deeper nodes are better because we need to get S_{t-1}
        return self.node.score > other.node.score  # Higher score is better


@dataclass
class ToTBFSState(Generic[StateT]):
    """State for Tree of Thoughts BFS algorithm."""

    tree: Tree[StateT]
    # Queue of (node, action) pairs to expand next
    next_nodes: List[Tuple[Node[StateT], str]] = dataclasses.field(default_factory=list)
    # Current depth level being processed
    current_depth: int = 0


class TreeOfThoughtsBFSAlgo(Algorithm[StateT, ToTBFSState[StateT]]):
    """
    Tree of Thoughts Breadth-First Search (ToT-BFS) algorithm.

    Original paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html

    This algorithm:
    1. Expands the tree breadth-first by depth level
    2. At each step, selects the b best-scoring leaf nodes of the same depth
    3. For each selected node, generates k new samples
    """

    def __init__(
        self,
        *,
        breadth_limit: int = 3,  # b: number of nodes to select at each step
        size_limit: int = 5,  # k: number of samples to generate for each node
    ):
        """
        Initialize the Tree of Thoughts BFS algorithm.

        Args:
            breadth_limit: Number of nodes to select at each step (b)
            size_limit: Number of samples to generate for each node (k)
        """
        assert size_limit > 0, "size_limit (k) should be greater than 0"
        assert breadth_limit > 0, "breadth_limit (b) should be greater than 0"
        assert size_limit >= breadth_limit, (
            "size_limit (k) should be greater than or equal to breadth_limit (b)"
        )

        self.size_limit = size_limit
        self.breadth_limit = breadth_limit

    def step(
        self,
        state: ToTBFSState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> ToTBFSState:
        """
        Run one step of the ToT-BFS algorithm, expanding a single node.

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
            # For the root node, just add it with the first action
            if not state.tree.root.children:
                action = next(iter(generate_fn))
                state.next_nodes.append((state.tree.root, action))
            else:
                # Otherwise, select best nodes at the deepest level
                self._queue_next_nodes(state, generate_fn)

        # If we still have nodes to expand
        if state.next_nodes:
            # Get the next node and action to expand
            node, action = state.next_nodes.pop(0)

            # Generate a new state
            new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost  = generate_fn[action](node)
            # Add to the tree
            state.tree.add_node(
                (new_state, new_score, new_scores, new_cost, new_total_cost, new_output_cost, new_total_output_cost, new_input_cost, new_total_input_cost, new_prm_output_cost, new_prm_total_output_cost, new_prm_input_cost, new_prm_total_input_cost), 
                node
            )
            # Update current depth
            state.current_depth = max(state.current_depth, node.depth + 1)

        return state

    def _queue_next_nodes(
        self, state: ToTBFSState, generate_fn: Mapping[str, GenerateFnType[StateT]]
    ) -> None:
        """
        Select the best nodes at the deepest level and queue them for expansion.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions
        """
        priority_queue: List[TreeOfThoughtsBFSHeapItem] = []

        # Find all unexpanded leaf nodes
        leaf_nodes = [
            node
            for node in state.tree.get_nodes()
            if not node.children and node != state.tree.root
        ]

        # If we have leaf nodes, find the deepest level
        max_depth = 0
        if leaf_nodes:
            max_depth = max(node.depth for node in leaf_nodes)
            state.current_depth = max_depth

        # Add all leaf nodes at the deepest level to the priority queue
        for node in leaf_nodes:
            if node.depth == max_depth:
                heappush(priority_queue, TreeOfThoughtsBFSHeapItem(node))

        # Select the top breadth_limit nodes
        selected_nodes = []
        for _ in range(min(self.breadth_limit, len(priority_queue))):
            if not priority_queue:
                break

            selected = heappop(priority_queue)
            selected_nodes.append(selected.node)

        # For each selected node, queue up expansions for all actions
        # Distribute the size_limit across all actions
        actions = list(generate_fn.keys())
        samples_per_action = max(1, self.size_limit // len(actions))

        for node in selected_nodes:
            # Queue expansions for each action
            for action in actions:
                for _ in range(samples_per_action):
                    state.next_nodes.append((node, action))

    def init_tree(self) -> ToTBFSState:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return ToTBFSState(tree=tree)

    def get_state_score_pairs(self, state: ToTBFSState) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()

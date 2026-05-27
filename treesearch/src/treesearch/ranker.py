from typing import List, Optional, TypeVar

from treesearch.algos.base import Algorithm
from treesearch.types import RankingFnType, StateScoreType

NodeStateT = TypeVar("NodeStateT")
AlgoStateT = TypeVar("AlgoStateT")


def top_k(
    state: AlgoStateT,
    algorithm: Algorithm,
    k: int,
    ranking_fn: Optional[RankingFnType[NodeStateT]] = None,
) -> List[StateScoreType[NodeStateT]]:
    """
    Returns the top-k states, given by the top-k elements after applying ranking_fn to all the state-score pairs.
    """
    state_scores = algorithm.get_state_score_pairs(state)
    if len(state_scores) < k:
        raise RuntimeError(
            f"The number of states generated so far is only {len(state_scores)}, and we cannot extract top-{k} elements from these states."
        )

    if ranking_fn is None:
        # If no ranking_fn is specified by users, we sort the answers by score values in descending order and return top-k.
        return sorted(state_scores, key=lambda x: x[1], reverse=True)[:k]
    else:
        return ranking_fn(state_scores)[:k]

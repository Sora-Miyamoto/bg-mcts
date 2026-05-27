from abc import ABC
from typing import List, Optional, Tuple

from bg_mcts.data_types import Action
from bg_mcts.llm_generation_interface import GenerationResult


class Task(ABC):
    def generate_eval_results(
        self, llm_answer: GenerationResult, kind: Action
    ) -> float:
        raise NotImplementedError()
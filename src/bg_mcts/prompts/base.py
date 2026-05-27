from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

from bg_mcts.data_types import Action
from bg_mcts.llm_generation_interface import GenerationRequest, GenerationResult
from bg_mcts.eval_result import EvalResult


class PromptTemplate(ABC):
    """
    Prompt Template for each task.
    The system prompt will **NOT** be handled by this class, and
    configuring system prompt is the responsibility of Model class constructor.
    """

    @abstractmethod
    def initial_prompt(self) -> str | List[str | Image.Image]:
        """
        Initial instruciton to be given to LLM.
        """
        raise NotImplementedError()

    @abstractmethod
    def continue_prompt(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def add_next_action_instruction(
        self, action: Action, next_prompt: GenerationRequest
    ) -> GenerationRequest:
        """
        Instruct LLMs to perform what action should be performed next.
        We can append and/or modify next_prompt e.g. appending
        the action instruction at the end of it.
        """
        raise NotImplementedError()

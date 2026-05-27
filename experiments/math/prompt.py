from bg_mcts.data_types import Action
from bg_mcts.llm_generation_interface import GenerationRequest
from bg_mcts.prompts.base import PromptTemplate
from bg_mcts.prompts.prompt_configs import PromptConfig
from bg_mcts.tasks.math500.task import Math500Problem


class BaselinePrompt(PromptTemplate):
    version = "baseline"

    def __init__(self, prompt_config: PromptConfig, data: Math500Problem):
        self.data = data

    def initial_prompt(self) -> str:
        prompt = initial_prompt(self.data.problem)
        return prompt
    
    def continue_prompt(self) -> str:
        prompt = "But wait, Let me think about the problem again.\n\nStep 1: "
        return prompt

    def add_next_action_instruction(
        self, action: Action, next_prompt: GenerationRequest
    ) -> GenerationRequest:
        return next_prompt


def initial_prompt(problem) -> str:
    #initial prompt = below
    # - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
    initial_prompt = f"""
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is the final number or expression in LaTeX format. Think step by step before answering.
Example: 
Example Problem:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Example Solution:
Step 1: Natalia sold 48 clips in April.

Step 2: In May, she sold half as many clips as in April. Half of 48 is 48 / 2 = 24 clips.

Step 3: To find the total number of clips sold in April and May, add the number of clips sold in each month: 48 + 24 = 72.

Step 4: Therefore the final answer is: $\\boxed{{72}}$. I hope it is correct.

Now, solve the following question: {problem}
"""
    return  initial_prompt

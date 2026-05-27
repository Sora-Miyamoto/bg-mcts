import json
import concurrent.futures
from pathlib import Path
from typing import List, Optional

from bg_mcts.data_types import Action, Math500ProbData
from bg_mcts.llm.PRM.gen_prm_vllm import GenPRM
from bg_mcts.llm_generation_interface import GenerationResult
from bg_mcts.tasks.base import Task
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

class Math500Problem(Task):
    def __init__(self, data: Math500ProbData, prm_name: str, prm_setting: dict) -> None:
        self.problem = data["problem"]
        self.answer = data["answer"]
        self.subject = data["subject"]
        self.level = data["level"]
        self.prm_name = prm_name
        
        if self. prm_name == "GenPRM/GenPRM-7B":
            self.GenPRM = GenPRM(prm_name=self.prm_name, problem=self.problem, prm_setting=prm_setting)
        else:
            print(f"PRM {self.prm_name} is not implemented. Please choose from ['GenPRM/GenPRM-7B']")
    
    @classmethod
    def load_file(cls, json_path: Path | str, prm_name: str, prm_setting: dict) -> "Math500Problem":
        prob_path = Path(json_path)
        if not prob_path.exists():
            raise RuntimeError(f"Math500 problem not found at {str(prob_path)}")

        return cls(data=json.loads(prob_path.read_text()), prm_name=prm_name, prm_setting=prm_setting)
        
    def run_with_timeout(self, func, kwargs=None, timeout=600, retry=3):
        if kwargs is None:
            kwargs = {}

        for attempt in range(1, retry + 1):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, **kwargs)
                try:
                    result = future.result(timeout=timeout)
                    return result
                except concurrent.futures.TimeoutError:
                    print(f"{attempt}times: timeout retry")
        print("all tries timeout")
        return None

    def generate_eval_results(
        self, llm_answer: GenerationResult, kind: Action, generation_state: Optional[List[str]]=None, parent_node=None, stopping_rule="process"
    ):
        normalization_config = NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed="all",
            equations=True,
        )

        extraction_config = LatexExtractionConfig(
            try_extract_without_anchor=True,
            boxed_match_priority=55,
            normalization_config=normalization_config,
        )
        if stopping_rule == "sequential":
            time_limit = 600
        elif stopping_rule == "full":
            time_limit = 3600

        if self.prm_name == "GenPRM/GenPRM-7B":
            input_messages = llm_answer.request.evaluation_messages
            result = None
            while result is None:
                result = self.run_with_timeout(
                    func=self.GenPRM.generate,
                    kwargs=dict(
                        messages=input_messages,
                        generation=llm_answer.generation,
                        parent_node=parent_node,
                        stopping_rule=stopping_rule
                    ),
                    timeout=time_limit,  
                    retry=2
                )
                if result is not None:
                    eval_result, eval_results, evaluation_messages, prm_output_cost, prm_input_cost = result
                    break
        else:
            raise NotImplementedError(f"PRM {self.prm_name} is not implemented. Please choose from ['GenPRM/GenPRM-7B']")
        
        if generation_state[0] == 'answered':
            boxed_answer = "\\boxed{" + self.answer + "}"
            new_generation = generation_state[1].replace("!", "")
            parsed_generation = parse(new_generation, extraction_config=[extraction_config])
            parsed_answer = parse(boxed_answer, extraction_config=[extraction_config])
            match_score = verify(parsed_generation, parsed_answer)
            if match_score == True:
                generation_state[0] = "correct"
                generation_state[1] = parsed_generation
            else:
                generation_state[0] = 'incorrect'
                generation_state[1] = parsed_generation
        
        return  eval_result, eval_results, generation_state, evaluation_messages, prm_output_cost, prm_input_cost

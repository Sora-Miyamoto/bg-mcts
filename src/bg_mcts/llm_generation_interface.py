from bg_mcts.model_base import GenerationRequest as BaseGenerationRequest
from bg_mcts.model_base import GenerationResult as BaseGenerationResult

class GenerationResult(BaseGenerationResult):

    def parse_math_answer(self):
        generation = self.generation
    
        if generation[-5:] == 'Step ':
            result = ['process', generation]
        else:
            result = ['answered', generation]
        return result


class GenerationRequest(BaseGenerationRequest):
    def get_last_generation_result(self) -> GenerationResult:
        assert self.messages[-1].role == "user" and len(self.messages) >= 3
        assert self.messages[-2].role == "assistant"

        return GenerationResult(
            request=GenerationRequest(messages=self.messages[:-2]),
            generation=self.messages[-2].content,
        )

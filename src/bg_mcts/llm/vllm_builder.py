from bg_mcts.llm.llm_interface import Model
from bg_mcts.llm.vllm.llama_vllm import LlamaVllm
from bg_mcts.llm.vllm.llama_vllm import PRICING as META_PRICING
from bg_mcts.llm.vllm.qwen_vllm import QwenVllm
from bg_mcts.llm.vllm.qwen_vllm import PRICING as QWEN_PRICING
from bg_mcts.llm.vllm.gemma_vllm import GemmaVllm
from bg_mcts.llm.vllm.gemma_vllm import PRICING as GEMMA_PRICING

def build_model(model_name: str, model_setting: dict, environment: str) -> Model:
    if model_name in  META_PRICING:
        model = LlamaVllm(model_name=model_name, model_setting=model_setting, environment=environment)
    elif model_name in QWEN_PRICING:
        model = QwenVllm(model_name=model_name, model_setting=model_setting, environment=environment)
    elif model_name in GEMMA_PRICING:
        model = GemmaVllm(model_name=model_name, model_setting=model_setting, environment=environment)
    else:
        raise ValueError(f"Unsupported model {model_name}")

    return model
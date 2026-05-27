from dotenv import load_dotenv
import copy
import re
from bg_mcts.llm.llm_interface import Model
from openai import OpenAI
import os
from transformers import AutoTokenizer


PRICING = {"Qwen/Qwen2.5-7B-Instruct": 1,
           "Qwen/Qwen3-8B": 1,
           "Qwen/Qwen3-14B": 1,
           "Qwen/Qwen3-32B": 1,
           }

class QwenVllm(Model):
    def __init__(
        self,
        model_name: str,     
        model_setting: dict,
        environment: str = "local",
        num_trial: int = 15,
    ) -> None:
        self.model_name = model_name
        self.model_setting = model_setting

        base_url = "http://localhost:" + str(model_setting["server_port"]) + "/v1"
        self.client = OpenAI(
            api_key="dummy",
            base_url=base_url,
            timeout=3600
        )
        self.num_trial = num_trial

        load_dotenv()
        FOLDER_PATH = os.getenv("FOLDER_PATH")
        HF_TOKEN = os.getenv("HF_TOKEN") 

        if model_name == "Qwen/Qwen2.5-7B-Instruct":
            chat_template_file = "qwen2_chat_template.jinja"
        elif model_name == "Qwen/Qwen3-8B":
            chat_template_file = "qwen3_chat_template.jinja"
        elif model_name == "Qwen/Qwen3-14B":
            chat_template_file = "qwen3_chat_template.jinja"
        elif model_name == "Qwen/Qwen3-32B":
            chat_template_file = "qwen3_chat_template.jinja"
        
        if environment == "local":
            chat_template_path = f"{FOLDER_PATH}src/bg_mcts/llm/vllm/{chat_template_file}"

        with open(chat_template_path, "r", encoding="utf-8") as f:
            custom_template = f.read()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        self.tokenizer.chat_template =custom_template

    def cut_messages(self, messages, max_tokens):
        max_model_len = int(self.model_setting["max_model_len"])
        where_cut = str(self.model_setting["where_cut"])

        ids = self.tokenizer.apply_chat_template(messages, tokenize=True)

        len_ids = len(ids)
        assistant_idx = 0
        if len_ids + max_tokens > max_model_len:
            if self.model_name == "Qwen/Qwen2.5-7B-Instruct":
                pattern = self.tokenizer.encode("<|im_start|>assistant\n") #[151644, 77091, 198]
            elif self.model_name == "Qwen/Qwen3-8B":
                pattern = self.tokenizer.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n") #[151644, 77091, 198, 151667, 271, 151668, 271]

            for i in range(len_ids):
                if ids[i:i+len(pattern)] == pattern:
                    assistant_idx = i + i+len(pattern)
                    break
            cut_tokens = len_ids + max_tokens - max_model_len + 150
            if where_cut == "middle":
                assistant_tokens = len(ids) - assistant_idx
                cut_start = int(((2 * assistant_tokens) / 5) + assistant_idx)
                if cut_tokens % 2 == 1:
                    new_ids = ids[:cut_start - int((cut_tokens + 1) / 2)] + [self.tokenizer.encode("...")[-1]] + ids[cut_start + int((cut_tokens + 1) / 2):]
                else:
                    new_ids = ids[:cut_start - int(cut_tokens / 2)] + [self.tokenizer.encode("...")[-1]] + ids[cut_start + int(cut_tokens / 2) + 1:]
            elif where_cut == "front":
                new_ids = ids[:assistant_idx] + [self.tokenizer.encode("...")[-1]] + ids[assistant_idx + 2 + cut_tokens:]
            new_text = self.tokenizer.decode(new_ids[assistant_idx:])
            new_messages = copy.deepcopy(messages)
            new_messages[-1]["content"] = new_text
            return new_messages
        else:
            return messages
            

    def generate(
        self, messages: list[dict[str, str]], stopping: bool, left_cost: float
    ) -> tuple[str, float]:
        if float(self.model_setting["max_tokens"]) > left_cost:
            max_tokens = int(left_cost)
        else:
            max_tokens = int(self.model_setting["max_tokens"])
        new_messages = self.cut_messages(messages, max_tokens)
        generation_state = ["", ""]
        if stopping:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=new_messages,
                temperature=float(self.model_setting["temperature"]),
                top_p=float(self.model_setting["top_p"]),
                stop=["\n\nStep", "\nStep"],
                max_tokens=max_tokens,
                extra_body={
                    "repetition_penalty": float(self.model_setting["repetition_penalty"]),
                    "include_stop_str_in_output": True,
                    "add_generation_prompt": False,
                },
            )
            generation = response.choices[0].message.content
            output_cost = float(response.usage.completion_tokens)
            if response.choices[0].finish_reason == "length":
                pattern = r"answer is(.*)\\boxed\{.*?\}"
                matches = re.search(pattern, generation)
                if matches:
                    generation_state = ["answered", generation]
                else:
                    generation_state = ["process", generation]
                    
                generation += "\n\nStep"
                generation_state = ["process", generation]
            elif response.choices[0].finish_reason == "stop":
                pattern = r"answer is(.*)\\boxed\{.*?\}"
                matches = re.search(pattern, generation)
                if matches:
                    generation_state = ["answered", generation]
                elif generation[-6:] == "\n\nStep" or generation[-5:] == "\nStep":
                    generation_state = ["process", generation]
                else:
                    generation_state = ["answered", generation]
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=new_messages,
                temperature=float(self.model_setting["temperature"]),
                top_p=float(self.model_setting["top_p"]),
                max_tokens=max_tokens,
                extra_body={
                    "repetition_penalty": float(self.model_setting["repetition_penalty"]),
                    "include_stop_str_in_output": True,
                    "add_generation_prompt": False,
                },
            )
            generation = response.choices[0].message.content
            output_cost = float(response.usage.completion_tokens)
            if response.choices[0].finish_reason == "length":
                pattern = r"answer is(.*)\\boxed\{.*?\}"
                matches = re.search(pattern, generation)
                if matches:
                    generation_state = ["answered", generation]
                else:
                    generation_state = ["process", generation]
        
            elif response.choices[0].finish_reason == "stop":
                generation_state = ["answered", generation]
        
        input_cost = float(len(self.tokenizer.apply_chat_template(new_messages, tokenize=True)))
        return generation, output_cost, input_cost, generation_state
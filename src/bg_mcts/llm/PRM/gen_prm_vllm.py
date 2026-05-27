from openai import OpenAI
from transformers import AutoTokenizer
import os
import sys
from dotenv import load_dotenv
import copy
import re
from bg_mcts.llm.llm_interface import Model

sys.path.append(os.path.dirname(__file__))
from gen_prm_evaluate_func import CodeExecutor, build_messages, generate_score_messages, cut_messages

class GenPRM(Model):
    def __init__(
        self,
        prm_name: str,
        problem: str,
        prm_setting: dict,
        num_trial: int = 15,
    ) -> None:
        self.prm_name = prm_name
        self.problem = problem

        load_dotenv()
        HF_TOKEN = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(self.prm_name, token=HF_TOKEN)

        self.num_trial = num_trial
        self.prm_setting = prm_setting

        base_url = "http://localhost:" + str(prm_setting["server_port"]) + "/v1"
        self.client = OpenAI(
            api_key="dummy",
            base_url=base_url,
            timeout=3600,
        )
        self.code_executor = CodeExecutor()
        
    def generate(
        self, messages: list[dict[str, str]], generation: str, parent_node=None, stopping_rule="sequential"
    ) -> float:
        if stopping_rule == "sequential":
            prm_input_cost = float(0)
            prm_output_cost = float(0)
        

            new_messages = copy.deepcopy(messages)
            if parent_node.state is None:
                cur_step = 1
                if generation[-4:] == "Step":
                    new_messages = build_messages("Step 1: " + generation[:-4], self.problem)
                else:
                    new_messages = build_messages("Step 1: " + generation, self.problem)
                input_messages = new_messages
            else:
                cur_step = parent_node.depth + 1
                parent_generation = parent_node.state.generation_result.generation
                if parent_node.state.generation_result.generation_state[0] != "process":
                    if generation[-4:] == "Step":
                        new_messages = build_messages("Step 1: " + generation[:-4], self.problem)
                    else:
                        new_messages = build_messages("Step 1: " + generation, self.problem)
                else:
                    if len(parent_generation) > 4:
                        if parent_generation[-4:] == "Step":
                            generation = "Step" + generation
                        if generation[-4:] == "Step":
                            generation = generation[:-4]
                    new_messages = copy.deepcopy(messages)
                    new_messages.append({"role": "user", "content": generation}) 
                input_messages = cut_messages(messages=new_messages, tokenizer=self.tokenizer, prm_setting=self.prm_setting)
            
            reward = float(0)
            for i in range(self.prm_setting["prm_call"]):
                output, reward_temp = generate_score_messages(
                    model_name= self.prm_name,
                    client=self.client,
                    tokenizer=self.tokenizer,
                    messages=input_messages,
                    cur_step=cur_step,
                    analyze=self.prm_setting["analyze"],
                    verify=self.prm_setting["verify"],
                    execute=self.prm_setting["execute"],
                    time_limit=int(self.prm_setting["time_limit"]),
                    temperature=float(self.prm_setting["temperature"]),
                    top_p=float(self.prm_setting["top_p"]),
                    top_k=int(self.prm_setting["top_k"]),
                    top_logprobs=int(self.prm_setting["top_logprobs"]),
                    max_tokens=int(self.prm_setting["max_tokens"]),
                    repetition_penalty=float(self.prm_setting["repetition_penalty"]),
                    vllm_seed=int(self.prm_setting["vllm_seed"]),
                    code_executor=self.code_executor,
                    logging=False,
                )
                reward += reward_temp
                prm_input_cost += float(len(self.tokenizer.apply_chat_template(input_messages, tokenize=True)))
                prm_output_cost += float(len(self.tokenizer.tokenize(output)))
            reward = reward / float(self.prm_setting["prm_call"])
            new_messages.append({"role": "assistant", "content": output})
            reward_list = [reward]

            return reward, reward_list, new_messages, prm_output_cost, prm_input_cost
        elif stopping_rule == "full":
            messages = []
            reward_list = []
            prm_input_cost = float(0)
            prm_output_cost = float(0)
            generation = "Step 1:" + generation
            generation_list = re.split(r'\n(?=Step)', generation)
            for i, item in enumerate(generation_list):
                cur_step = i + 1
                if i == 0:
                    messages = build_messages(generation_list[0], self.problem)
                    input_messages = messages
                    reward = float(0)
                    for i in range(self.prm_setting["prm_call"]):
                        output, reward_temp = generate_score_messages(
                            model_name= self.prm_name,
                            client=self.client,
                            tokenizer=self.tokenizer,
                            messages=input_messages,
                            cur_step=cur_step,
                            analyze=self.prm_setting["analyze"],
                            verify=self.prm_setting["verify"],
                            execute=self.prm_setting["execute"],
                            time_limit=int(self.prm_setting["time_limit"]),
                            temperature=float(self.prm_setting["temperature"]),
                            top_p=float(self.prm_setting["top_p"]),
                            top_k=int(self.prm_setting["top_k"]),
                            top_logprobs=int(self.prm_setting["top_logprobs"]),
                            max_tokens=int(self.prm_setting["max_tokens"]),
                            repetition_penalty=float(self.prm_setting["repetition_penalty"]),
                            vllm_seed=int(self.prm_setting["vllm_seed"]),
                            code_executor=self.code_executor,
                            logging=False,
                        )
                        reward += reward_temp
                        prm_input_cost += float(len(self.tokenizer.apply_chat_template(input_messages, tokenize=True)))
                        prm_output_cost += float(len(self.tokenizer.tokenize(output)))
                    reward = reward / float(self.prm_setting["prm_call"])
                    messages.append({"role": "assistant", "content": output})
                    reward_list.append(reward)
                else:
                    messages.append({"role": "user", "content": generation_list[i]})
                    new_messages = copy.deepcopy(messages)
                    input_messages = cut_messages(messages=new_messages, tokenizer=self.tokenizer, prm_setting=self.prm_setting)
                    reward = float(0)
                    for i in range(self.prm_setting["prm_call"]):
                        output, reward_temp = generate_score_messages(
                            model_name= self.prm_name,
                            client=self.client,
                            tokenizer=self.tokenizer,
                            messages=input_messages,
                            cur_step=cur_step,
                            analyze=self.prm_setting["analyze"],
                            verify=self.prm_setting["verify"],
                            execute=self.prm_setting["execute"],
                            time_limit=int(self.prm_setting["time_limit"]),
                            temperature=float(self.prm_setting["temperature"]),
                            top_p=float(self.prm_setting["top_p"]),
                            top_k=int(self.prm_setting["top_k"]),
                            top_logprobs=int(self.prm_setting["top_logprobs"]),
                            max_tokens=int(self.prm_setting["max_tokens"]),
                            repetition_penalty=float(self.prm_setting["repetition_penalty"]),
                            vllm_seed=int(self.prm_setting["vllm_seed"]),
                            code_executor=self.code_executor,
                            logging=False,
                        )
                        reward += reward_temp
                        prm_input_cost += float(len(self.tokenizer.apply_chat_template(input_messages, tokenize=True)))
                        prm_output_cost += float(len(self.tokenizer.tokenize(output)))
                    reward = reward / float(self.prm_setting["prm_call"])
                    messages.append({"role": "assistant", "content": output})
                    reward_list.append(reward)
            if reward_list:
                if str(self.prm_setting["scoring"]) == "average":
                    reward = sum(reward_list) / len(reward_list)
                elif str(self.prm_setting["scoring"]) == "last":
                    reward = reward_list[-1]
                elif str(self.prm_setting["scoring"]) == "min":
                    reward = min(reward_list)
                elif str(self.prm_setting["scoring"]) == "max":
                    reward = max(reward_list)
            else:
                reward = 0
                reward_list.append(reward)

            return reward, reward_list, messages, prm_output_cost, prm_input_cost

import math
import re
from contextlib import redirect_stdout
import io
import signal
import copy

def build_messages(generation, problem):
    messages = [
        {"role": "system", "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."},
        {"role": "user", "content": f"Question: {problem}\n\n{generation}"}
    ]
    return messages


def cut_messages(messages, tokenizer, prm_setting):
    max_model_len = int(prm_setting["max_model_len"])
    max_tokens = int(prm_setting["max_tokens"])

    new_messages = copy.deepcopy(messages)
    turns_messages = len(new_messages) / 2 - 1
    tokenized_messages = tokenizer.apply_chat_template(new_messages, tokenize=True, add_generation_prompt=False)
    tokens_messages = len(tokenized_messages)
    
    if tokens_messages + 100 <= max_model_len - max_tokens:
        return new_messages
    else:
        turns_messages = int(len(new_messages) / 2 - 1)
        if turns_messages % 2 == 1:
            cut_index = int(turns_messages + 1)
            cut_step = int(cut_index / 2)
        else:
            cut_index = int(turns_messages)
            cut_step = int(cut_index / 2)
        count = 0

        while tokens_messages + 100 > max_model_len - max_tokens:
            assistant_content = new_messages[cut_index]["content"]
            tokenized_assistant = tokenizer.tokenize(assistant_content)
            if tokenized_assistant[-6] == "Yes":
                new_messages[cut_index]["content"] = "<analyze>\nLet's analyze the Paragraph " + str(cut_step) + " step by step: ..... </analyze>\n<output>\n**Judgement**: $\\boxed{Yes}$\n</output>\n"
            else:
                new_messages[cut_index]["content"] = "<analyze>\nLet's analyze the Paragraph " + str(cut_step) + " step by step: ..... </analyze>\n<output>\n**Judgement**: $\\boxed{No}$\n</output>\n"
            count += 1
            if count % 2 == 1:
                cut_index -= int(2 * count)
                cut_step = int(cut_index / 2)
            else:
                cut_index += int(2 * count)
                cut_step = int(cut_index / 2)
            tokenized_messages = tokenizer.apply_chat_template(new_messages, tokenize=True, add_generation_prompt=False)
            tokens_messages = len(tokenized_messages)
            if turns_messages % 2 == 0 and cut_index == 0:
                cut_index = len(new_messages) - 2
                cut_step = turns_messages
            if cut_index >= (len(new_messages) - 1) or cut_index < 0:
                print("Error: We can not cut more")
                break

        if turns_messages % 2 == 1:
            cut_index = int(turns_messages)
            cut_step = int((cut_index + 1) / 2)
        else:
            cut_index = int(turns_messages - 1)
            cut_step = int((cut_index + 1) / 2)
        count = 0
        while tokens_messages + 100 > max_model_len -  max_tokens:
            user_content = new_messages[cut_index]["content"]
            if user_content.startswith("Step"):
                match = re.search(r"Step.*?:", user_content)
                if match:
                    cut_start = match.end() -1
                else:
                    cut_start = 10
            elif user_content.startswith("But wait, Let me think about the problem again.\n\nStep 1: "):
                cut_start = 55
            else:
                cut_start = 10

            new_messages[cut_index]["content"] = user_content[:cut_start+1] + "..."
            count += 1
            if count % 2 == 1:
                cut_index -= int(2 * count)
                cut_step = int((cut_index + 1) / 2)
            else:
                cut_index += int(2 * count)
                cut_step = int((cut_index + 1) / 2)
            tokenized_messages = tokenizer.apply_chat_template(new_messages, tokenize=True, add_generation_prompt=False)
            tokens_messages = len(tokenized_messages)

            if turns_messages % 2 == 0 and cut_index == -1:
                cut_index = len(messages) - 3
                cut_step = turns_messages
            if cut_index >= (len(new_messages) - 1 ) or cut_index < -1:
                print("Error: We can not cut more")
                break
            
        return new_messages


       
def cprint(s, start):
    if not isinstance(s, str):
        s = str(s)

    print(f"{'*' * 40}")
    print(f"Start: {start}")
    print(f"{'-' * 40}")

    print(s.replace('\n', '\\n'))

    print(f"{'-' * 40}")
    print(f"End: {start}")
    print(f"{'*' * 40}\n")

class timeout:
    """timeout context manager"""

    def __init__(self, seconds=1):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)

    def handle_timeout(self, signum, frame):
        raise TimeoutError("Code execution timed out")

class CodeExecutor:
    """code executor"""

    def __init__(self):
        self.namespace = {}  # indicate the global namespace for exec
        self.code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)

    def execute(self, text):
        # extract code block
        try:
            code_block = self.code_pattern.findall(text)[-1].strip()
        except Exception as e:
            actual = f"Code format error: No code found."
            return actual

        # execute code block
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                with timeout(seconds=5):
                    exec(code_block, self.namespace)
            actual = f.getvalue().strip()
        except TimeoutError as te:
            actual = f"Code execute time out: {te}"
            print(actual)
        except Exception as e:
            actual = f"Code execute Error: {type(e).__name__}: {e}"
            print(actual)

        return actual

def get_reward_score_vllm(out): # out = response.choices[0]
        '''calculate the reward score'''
        generated_text = out.message.content
        logprobs = out.logprobs.content
        
        # find the position of Yes/No token
        # boxed_match = re.search(r'(Yes|No)\}', generated_text)
        boxed_match = re.findall(r'(Yes|No)\}', generated_text)
    
        if boxed_match:
            decision = boxed_match[-1].capitalize()
            if decision == "Yes":
                # convert logprob to probability
                for item in reversed(logprobs):
                    if item.token.strip() == "Yes":
                        judged_logrobs = item
                        yes_logprob = judged_logrobs.logprob
                        yes_prob = math.exp(yes_logprob) 
                        break
                try:
                    for item in judged_logrobs.top_logprobs:
                        if item.token.strip() == 'No':
                            no_logprob = item.logprob
                            break
                    no_prob = math.exp(no_logprob)
                except NameError:
                    min_logprob = min(item.logprob for item in judged_logrobs.top_logprobs)
                    no_prob = math.exp(min_logprob)
                # calculate softmax value
                softmax_denominator = yes_prob + no_prob

                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator
                return softmax_yes
            
            elif decision == "No":
                
                for item in reversed(logprobs):
                    if item.token.strip() == "No":
                        judged_logrobs = item
                        no_logprob = judged_logrobs.logprob
                        no_prob = math.exp(no_logprob) 
                        break
                try:
                    for item in judged_logrobs.top_logprobs:
                        if item.token.strip() == 'Yes':
                            yes_logprob = item.logprob
                            break
                    yes_prob = math.exp(yes_logprob)
                except NameError:
                    # set 'No' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(item.logprob for item in judged_logrobs.top_logprobs)
                    yes_prob = math.exp(min_logprob)
                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator
                return softmax_yes
        else:
            return 0.5

def generate_score_messages(
        model_name,
        client,         
        tokenizer,
        messages,
        cur_step=1,
        analyze=True,
        verify=True,
        execute=True,
        time_limit=3,
        temperature=0.6,
        top_p = 0.95,
        top_k = 20,
        top_logprobs = 20,
        max_tokens=4096,
        repetition_penalty = 1.0,
        vllm_seed = 1,
        code_executor=None,
        analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
        verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
        output_template="<output>\n**Judgement**: $\\boxed",
        logging=True
    ):

    context = {"cur_step": cur_step}
    analyze_start = analyze_template.format(**context)
    verify_start = verify_template.format(**context)
    output_start = output_template.format(**context)

    # make a evaluation message
    new_messages = copy.deepcopy(messages)
    # Stage 1: Analyze
    if analyze:
        if logging:
            cprint(messages[1]["content"] + analyze_start, f'paragraph {cur_step} request 1')
        new_messages = copy.deepcopy(messages)
        new_messages[-1]["content"] = new_messages[-1]["content"] + analyze_start
        response = client.chat.completions.create(
            model=model_name,
            messages=new_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
            stop=['</analyze>\n'],
            seed=vllm_seed,
            extra_body={
                "top_k": top_k, 
                "repetition_penalty": repetition_penalty,
                "include_stop_str_in_output": True,
                "add_generation_prompt": False,
            },
        )
        if verify:
            cur_prompt = analyze_start + response.choices[0].message.content + verify_start
        else:
            cur_prompt = analyze_start + response.choices[0].message.content + output_start

    elif verify:
        cur_prompt = verify_start
    else:
        cur_prompt = output_start
        
    # Stage 2: Verify / Output loop
    cur_prompts = [cur_prompt]
    out_nodes = []
    cur_time = 0

    while len(cur_prompts) > 0:
        tokenized_prompt = tokenizer.tokenize(cur_prompts[0])
        left_tokens = max_tokens - len(tokenized_prompt)
        if left_tokens > 0 and cur_time < time_limit:
            if logging:
                cprint(cur_prompts[0], f'paragraph {cur_step} request {cur_time + 2}')

            if verify and execute:
                stop_strings = ['\n```\n', '</output>\n']
            else:
                stop_strings = ['</output>\n']
            new_messages[-1]["content"] = messages[-1]["content"] + cur_prompts[0]
            response2 = client.chat.completions.create(
                model=model_name,
                messages=new_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=left_tokens,
                logprobs=True,
                top_logprobs=top_logprobs,
                stop=stop_strings,
                seed=vllm_seed,
                extra_body={
                    "top_k": top_k, 
                    "repetition_penalty": repetition_penalty,
                    "include_stop_str_in_output": True,
                    "add_generation_prompt": False,
                },
            )
        else:
            # if the time limit is reached, or the left tokens are not enough
            if analyze:
                # degrade into analyze mode
                cur_prompts = [analyze_start + response.choices[0].message.content.split('</analyze>')[0] + '</analyze>\n' + output_start]
            else:
                # enter the output mode
                cur_prompts = [cur_prompts[0] + '</verify>\n' + output_start]
            tokenized_prompt = tokenizer.tokenize(cur_prompts[0])
            left_tokens = 20

            if logging:
                cprint(cur_prompts[0], f'paragraph {cur_step} request {cur_time + 2}')

            new_messages[-1]["content"] = messages[-1]["content"] + cur_prompts[0]

            response2 = client.chat.completions.create(
                model=model_name,
                messages=new_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=left_tokens,
                logprobs=True,
                top_logprobs=top_logprobs,
                stop=['</output>\n'],
                seed=vllm_seed,
                extra_body={
                    "top_k": top_k, 
                    "repetition_penalty": repetition_penalty,
                    "include_stop_str_in_output": True,
                    "add_generation_prompt": False,
                },
            )
        cur_time += 1
        new_prompts = []

        if response2.choices[0].message.content.endswith('</output>\n'):
            response2.choices[0].message.content = cur_prompts[0] + response2.choices[0].message.content
            out_nodes.append(response2)
        else:
            if execute:
                code_output = code_executor.execute(cur_prompts[0] + response2.choices[0].message.content)
                code_content = f"[Code Output]\n\n```\n{code_output}\n```\n"
                new_prompts.append(cur_prompts[0] + response2.choices[0].message.content + code_content)
            else:
                new_prompts.append(cur_prompts[0] + response2.choices[0].message.content + '[Code Output]\n\n```\n')

        cur_prompts = new_prompts

    output2 = out_nodes[0]
    reward_score = get_reward_score_vllm(output2.choices[0])

    return output2.choices[0].message.content, reward_score

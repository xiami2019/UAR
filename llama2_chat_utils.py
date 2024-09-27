# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
from typing import List, Literal, TypedDict


Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                ) + [tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens

def format_tokens_triviaqa(dialogs, tokenizer):
    batched_input = []
    for dialog in dialogs:
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        batched_input.append(f"{B_INST} {(dialog).strip()} {E_INST}")
    
    batched_input = tokenizer(
        batched_input,
        return_tensors="pt",
        padding=True,
    )

    for k in batched_input:
        batched_input[k] = batched_input[k].cuda()
        
    return batched_input

def format_tokens_uar(data, tokenizer, data_type):
    zero_shot_prompt_chat = "[INST] {} [/INST]".format

    zero_shot_promp_drop = '''[INST] Please answer the question based on the given passage. 
Passage: {}
Question: {}
Now give me the answer. [/INST]'''.format

    zero_shot_gsm8k = '''[INST] Answer the math word question step by step. Your answer needs to end with 'The answer is'.
Question: {}
Let's think step by step and give me the answer. [/INST]'''.format

    batched_input = []
    for single_data in data:
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        if data_type == 'normal':
            batched_input.append(zero_shot_prompt_chat(single_data['question']))
        elif data_type == 'drop':
            batched_input.append(zero_shot_promp_drop(single_data['passage'], single_data['question']))
        elif data_type == 'gsm8k':
            batched_input.append(zero_shot_gsm8k(single_data['question']))
        else:
            raise ValueError()
    
    batched_input = tokenizer(
        batched_input,
        return_tensors="pt",
        padding=True,
    )

    for k in batched_input:
        batched_input[k] = batched_input[k].cuda()
        
    return batched_input

def format_tokens_triviaqa_for_ppo_reward(dialogs, tokenizer, refuse_answer=None):
    batched_input = []
    for dialog in dialogs:
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.generated_answer
        """
        if refuse_answer is not None:
            batched_input.append(f"{B_INST} {dialog['question']} {E_INST} {refuse_answer}")
            # batched_input.append(f"{B_INST} {dialog['question']} {E_INST} {dialog['negative_answer']}")
        else:
            batched_input.append(f"{B_INST} {dialog['question']} {E_INST} {dialog['generated_answer']}")
            # batched_input.append(f"{B_INST} {dialog['question']} {E_INST} {dialog['positive_answer']}")

    batched_input = tokenizer(
        batched_input,
        return_tensors="pt",
        padding=True,
    )

    for k in batched_input:
        batched_input[k] = batched_input[k].cuda()
        
    return batched_input


def read_dialogs_from_file(file_path):
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as file:
            dialogs = [json.loads(line.strip()) for line in file.readlines()]
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            dialogs = json.load(file)
    return dialogs
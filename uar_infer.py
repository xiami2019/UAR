import fire
import os
import math
import json
import copy

import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel
from tqdm import tqdm
from torch.nn.functional import softmax

from llama2_chat_utils import read_dialogs_from_file, format_tokens_uar


# Function to load the main model for text generation
def load_model(model_name):
    model = LlamaForCausalLM.from_pretrained(model_name).cuda()
    return model

def load_value_head(value_head_name, config):
    partial_state_dict = torch.load(value_head_name)
    scorer = nn.Linear(config.hidden_size, 2, bias=False)
    
    scorer.weight.data = partial_state_dict['score.weight'].cuda()
    return scorer.cuda()

def cls_infer(transformer_outputs, value_head, input_ids, pad_token_id):
    hidden_states = transformer_outputs[0]
    logits = value_head(hidden_states)
    batch_size = input_ids.shape[0]

    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    sequence_lengths = sequence_lengths.to(logits.device)

    pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    return pooled_logits

def main(
    model_name: str=None,
    batch_size: int=1,
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    save_name: str = 'results.json',
    data_type: str = 'normal', # data_type = normal / drop / gsm8k
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs= read_dialogs_from_file(prompt_file)

    print(f"User dialogs number: {len(dialogs)}")
    print("\n==================================\n")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    base_model = LlamaModel.from_pretrained(model_name).cuda()

    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # load_value_head
    if '7b-chat' in model_name:
        intent_aware_head = load_value_head('classifiers/7b/Intent_aware_llama2_7b_chat/intent_cls_7b.pth', base_model.config)
        knowledge_aware_head = load_value_head('classifiers/7b/Know_aware_llama2_7b_chat/know_cls_7b.pth', base_model.config)
        time_aware_head = load_value_head('classifiers/7b/Time_aware_llama2_7b_chat/time_cls_7b.pth', base_model.config)
        self_aware_head = load_value_head('classifiers/7b/Self_aware_llama2_7b_chat/self_cls_7b.pth', base_model.config)
    elif '13b-chat' in model_name:
        intent_aware_head = load_value_head('classifiers/13b/Intent_aware_llama2_13b_chat/intent_cls_13b.pth', base_model.config)
        knowledge_aware_head = load_value_head('classifiers/13b/Know_aware_llama2_13b_chat/know_cls_13b.pth', base_model.config)
        time_aware_head = load_value_head('classifiers/13b/Time_aware_llama2_13b_chat/time_cls_13b.pth', base_model.config)
        self_aware_head = load_value_head('classifiers/13b/Self_aware_llama2_13b_chat/self_cls_13b.pth', base_model.config)

    all_value_heads = {
        'intent_aware_head': intent_aware_head,
        'knowledge_aware_head': knowledge_aware_head,
        'time_aware_head': time_aware_head,
        'self_aware_head': self_aware_head,
    }
    
    batch_num = math.ceil(len(dialogs) / batch_size)

    with torch.no_grad():

        current_generated_results = []
        for i in tqdm(range(batch_num)):
            chunk_data = copy.deepcopy(dialogs[i*batch_size:(i+1)*batch_size])
            input_data = [item for item in chunk_data]
            inputs = format_tokens_uar(input_data, tokenizer, data_type)
            with torch.no_grad():
                transformer_outputs = base_model(**inputs)
            inputs_id = inputs['input_ids']
            intent_logits = cls_infer(transformer_outputs, all_value_heads['intent_aware_head'], inputs_id, tokenizer.pad_token_id)
            knowledge_logits = cls_infer(transformer_outputs, all_value_heads['knowledge_aware_head'], inputs_id, tokenizer.pad_token_id)
            time_logits = cls_infer(transformer_outputs, all_value_heads['time_aware_head'], inputs_id, tokenizer.pad_token_id)
            self_logits = cls_infer(transformer_outputs, all_value_heads['self_aware_head'], inputs_id, tokenizer.pad_token_id)
            intent_logits = softmax(intent_logits, dim=1)
            knowledge_logits = softmax(knowledge_logits, dim=1)
            time_logits = softmax(time_logits, dim=1)
            self_logits = softmax(self_logits, dim=1)
            for idx in range(len(chunk_data)):
                intent_ce_reward_score = round(intent_logits[idx].cpu().numpy().tolist()[0], 4)
                knowledge_ce_reward_score = round(knowledge_logits[idx].cpu().numpy().tolist()[0], 4)
                time_ce_reward_score = round(time_logits[idx].cpu().numpy().tolist()[0], 4)
                self_ce_reward_score = round(self_logits[idx].cpu().numpy().tolist()[0], 4)

                # UAR Criteria
                if intent_ce_reward_score >= 0.5: # intent-aware-head
                    chunk_data[idx]['need_retrieve_predicted'] = True
                else:
                    if knowledge_ce_reward_score >= 0.5: # knowledge-aware-head
                        chunk_data[idx]['need_retrieve_predicted'] = False
                    else:
                        if time_ce_reward_score >= 0.5: # time-aware-head
                            chunk_data[idx]['need_retrieve_predicted'] = True
                        else:
                            if self_ce_reward_score >= 0.5: # self-aware-head
                                chunk_data[idx]['need_retrieve_predicted'] = True
                            else:
                                chunk_data[idx]['need_retrieve_predicted'] = False
            
            current_generated_results.extend(chunk_data)

    # save results
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(current_generated_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(main)

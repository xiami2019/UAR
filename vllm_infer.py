import os
import json
import torch
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm

B_INST, E_INST = "[INST]", "[/INST]"

def get_args():
    parser = argparse.ArgumentParser(description="Inferencing with VLLM")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--input_file", type=str, help="Input file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--data_type", type=str, default='normal', choices=['normal', 'gsm8k', 'drop'])
    parser.add_argument("--doc_num", type=int, default=5)

    return parser.parse_args()

def load_data(input_file):
    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            data = json.load(f)
    elif input_file.endswith(".jsonl"):
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]
    return data

if __name__ == '__main__':
    args = get_args()
    sampling_params = SamplingParams(top_k=1, max_tokens=2048) # greedy decoding
    llm = LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count())
    test_data = load_data(args.input_file)
    
    zero_shot_prompt_chat = "[INST] {} [/INST]".format

    zero_shot_ret_chat = '''{}

Here are some additional reference passages:
{}

You can refer to the content of relevant reference passages to answer the questions.
Now give me the answer.
'''.format

    zero_shot_ret_chat_gsm8k = '''Answer the math word question step by step. Your answer needs to end with 'The answer is'
Question: {}
    
Here are some additional reference passages:
{}

You can refer to the content of relevant reference passages to answer the questions.
Let's think step by step and give me the answer.
'''.format

    zero_shot_ret_chat_drop = '''Please answer the question based on the given passage. 
Passage: {}
Question: {}
    
Here are some additional reference passages:
{}

You can refer to the content of relevant reference passages to answer the questions.
Now give me the answer.
'''.format


    prompts = []
    for item in test_data:
        if item['need_retrieve_predicted']:
            question = item['question']
            if 'cotriever_results' in item:
                retrieve_results = [item['text'] for item in item['cotriever_results']]
                retrieve_results = retrieve_results[:args.doc_num]
            else:
                retrieve_results = [item['refer_passage']]
            
            ret_results = ''
            for idx, ret in enumerate(retrieve_results):
                ret_results += f"{idx}. {ret}\n"
            if args.data_type == 'gsm8k':
                input_prompt = zero_shot_ret_chat_gsm8k(question, ret_results)
            elif args.data_type == 'drop':
                passage = item['passage']
                input_prompt = zero_shot_ret_chat_drop(passage, question, ret_results)
            else:
                input_prompt = zero_shot_ret_chat(question, ret_results)
            prompts.append(zero_shot_prompt_chat(input_prompt))
        else:
            prompts.append(zero_shot_prompt_chat(item['question'].strip()))

    
    outputs = llm.generate(prompts, sampling_params)
    if args.data_type == 'drop':
        outputs_to_save = {}
        for output, sample in zip(outputs, test_data):
            outputs_to_save[sample['query_id']] = output.outputs[0].text.strip()
        with open(args.output_file, 'w') as f:
            json.dump(outputs_to_save, f, indent=2, ensure_ascii=False)
    else:
        for output, sample in zip(outputs, test_data):
            sample['generated_answer'] = output.outputs[0].text.strip()
    
        with open(args.output_file, 'w') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
    
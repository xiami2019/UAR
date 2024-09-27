from GPTWrapper import GPTWrapper
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial

input_text = '''In the following task, you are given a Question, a model Prediction for the Question, and a Ground-truth Answer to the Question. You should decide whether the model Prediction implies the Ground-truth Answer.

Question:
{}

Prediction:
{}

Ground-truth Answer:
{}
Does the Prediction imply the Ground-truth Answer? Output Yes or No:'''.format

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo-instruct', help='The engine to use.')
    parser.add_argument('--input_file', type=str, required=True, help='The input file to use.')
    parser.add_argument('--output_file', type=str, default=None, help='The output file to use.')
    parser.add_argument('--batch_size', type=int, default=20, help='The batch size to use.')
    parser.add_argument('--config_path', type=str, default='evaluations/api_keys_config.json', help='The config path to use.')
    parser.add_argument('--only_cal_acc', action='store_true', help='Only calculate accuracy.')
    args = parser.parse_args()

    return args

def process_batch(wrapper, engine, inputs_data, batch_size):
    prompts = []
    for sample in inputs_data:
        ground_truth_answer = ''
        ans_count = 0
        if 'answer' in sample:
            answers = sample['answer']
        elif 'answers' in sample:
            answers = sample['answers']
        else:
            answers = []
        
        for ans in answers:
            if ans != '':
                ans_count += 1
                ground_truth_answer += f"{ans_count}. {ans}\n"
        prompts.append(input_text(sample['question'], sample['generated_answer'], ground_truth_answer))

    if engine == 'gpt-3.5-turbo-0125':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompts[0]}
        ]
    else:
        messages = prompts

    responses = wrapper.completions_with_backoff(
        engine=engine, 
        messages=messages, 
        temperature=0.0, 
        max_tokens=50, 
        top_p=0.9, 
        frequency_penalty=0.1, 
        presence_penalty=0.1,
    )

    if batch_size == 1:
        responses = parse_response([responses])
    else:
        responses = parse_response(responses)

    for item, response in zip(inputs_data, responses):
        item['judge'] = response

    results = inputs_data[:]

    return results

def parse_response(responses):
    results = []
    for response in responses:
        if 'yes' in response.lower():
            results.append('yes')
        elif 'no' in response.lower():
            results.append('no')
        else:
            results.append('')
    return results

def update_progress(result):
    results.extend(result)
    pbar.update()

def compute_acc(results):
    correct_count = sum(1 for res in [item['judge'] for item in results] if res == 'yes')
    print(f"Correct: {correct_count}, Total: {len(results)}, Accuracy: {correct_count/len(results):.4f}")

if __name__ == '__main__':
    args = get_config()

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    if args.only_cal_acc:
        compute_acc(data)
        exit()

    assert args.engine is not None, 'Engine is required'
    assert args.output_file is not None, 'Output file is required'

    if args.engine == 'gpt-3.5-turbo-0125':
        assert args.batch_size == 1

    wrapper = GPTWrapper(
        config_path=args.config_path,
        base_wait_time=30,
    )
    
    results = []
    batch_size = args.batch_size
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    process_with_args = partial(process_batch, wrapper, args.engine, batch_size=batch_size)
    with Manager() as manager:
        with Pool(processes=128) as pool:
            with tqdm(total=len(batches)) as pbar:
                for batch in batches:
                    pool.apply_async(process_with_args, args=(batch,), callback=update_progress)
                pool.close()
                pool.join()

    compute_acc(results)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
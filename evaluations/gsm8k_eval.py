import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    file_name = args.file_name

    with open(file_name, 'r') as f:
        gsm8k_data = json.load(f)

    correct_count = 0
    wrong_format = 0
    for item in gsm8k_data:
        ground_truth_idx = item['answer'].index('####')
        ground_truth = item['answer'][ground_truth_idx+len('####'):].strip()
        response = item['generated_answer'].lower()
        if 'the answer is' not in response:
            predicted_answer = ''
            wrong_format += 1
        else:
            predicted_idx = response.index('the answer is')
            predicted_answer = response[predicted_idx+len('the answer is'):].strip()
        
        if ground_truth in predicted_answer:
            correct_count += 1
    print(correct_count / (len(gsm8k_data)))
    print('wrong format:', wrong_format)

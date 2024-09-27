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
        data = json.load(f)

    correct_count = 0
    for sample in data:
        correct_flag = False
        if 'answer' in sample:
            answers = sample['answer']
        elif 'answer_ground_truth' in sample:
            answers = sample['answer_ground_truth']
        elif 'answers' in sample:
            answers = sample['answers']
        else:
            raise ValueError()
        for ans in answers:
            if 'strategyqa' in file_name.lower():
                if sample['generated_answer'].lower().startswith(ans.lower()):
                    correct_flag = True
                    break
            else:
                if ans.lower() in sample['generated_answer'].lower():
                    correct_flag = True
                    break
        if correct_flag:
            correct_count += 1

    print(f"Correct: {correct_count}, Total: {len(data)}, Accuracy: {correct_count/len(data):.4f}")
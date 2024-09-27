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

    total_correct_count = 0
    results = {}
    labels = {'overall': []}
    predict = {'overall': []}
    for item in data:
        if item['scenario'] not in results:
            results[item['scenario']] = {'correct': 0, 'total': 0}
            labels[item['scenario']] = []
            predict[item['scenario']] = []
        results[item['scenario']]['total'] += 1

        if item['need_retrieve'] == True:
            labels[item['scenario']].append(1)
            labels['overall'].append(1)
        else:
            labels[item['scenario']].append(0)
            labels['overall'].append(0)

        if item['need_retrieve_predicted'] == True:
            predict[item['scenario']].append(1)
            predict['overall'].append(1)
        else:
            predict[item['scenario']].append(0)
            predict['overall'].append(0)


        if item['need_retrieve'] == item['need_retrieve_predicted']:
            total_correct_count += 1
            results[item['scenario']]['correct'] += 1

    for k in results:
        print('Scenario {} Acc: {:.2f};'.format(k, 100 * (results[k]['correct'] / results[k]['total'])))

    print('Overall Acc: {:.2f};'.format(100 * (total_correct_count / len(data))))

    print(results)
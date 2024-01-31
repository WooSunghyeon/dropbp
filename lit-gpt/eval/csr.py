import json
from pathlib import Path

def CSR_accuracy(json_file:Path,):
    with open(json_file, 'r') as file:
        data = json.load(file)
    total_acc = 0
    task_count = 0
    for test, values in data['results'].items():
        acc_value = values.get('acc_norm', values.get('acc'))
        print(f"{test}: {acc_value*100:.1f}")
        total_acc += acc_value
        task_count += 1
    if task_count > 0:
        average_acc = total_acc / task_count
        print(f"Average: {average_acc*100:.1f}")

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(CSR_accuracy, as_positional=False)


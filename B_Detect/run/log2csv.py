import os
import re
import csv
import argparse
import numpy as np

pattern_acc = re.compile(r"acc5: ([\d.]+).*acc1: ([\d.]+)")
pattern_subject = re.compile(r"subject:(S\d+)")
pattern_Namespace = re.compile(r"Namespace\((.*?)\)")

parser = argparse.ArgumentParser(description='exp_log2csv')
parser.add_argument('--today', type=str, default='2024-06-20', help='date')
parser.add_argument('--seeds', nargs='+', type=int, default=[2024, 2025, 2026, 2027, 2028, 2029], help='random seed')
parser.add_argument('--subjects', nargs='+', type=str, default=['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08'], help='subjects')
args = parser.parse_args()

today = args.today
seeds = args.seeds
subjects = args.subjects

print('today:', today)
print('seeds:', seeds)
print('subjects:', subjects)

# root
log_root_dir = '../log_save/' + today

all_data = {subject: {seed: [] for seed in seeds} for subject in subjects}
all_Namespace = {seed: [] for seed in seeds}

for seed in seeds:
    log_dir = os.path.join(log_root_dir, str(seed))
    log_file = os.path.join(log_dir, 'record.log')
    
    if not os.path.exists(log_file):
        print(f"Log file for seed {seed} does not exist.")
        continue
    
    with open(log_file, 'r') as file:
        log_lines = file.readlines()
    
    current_subject = args.subjects[0]
    for line in log_lines:
        subject_match = pattern_subject.search(line)
        if subject_match:
            current_subject = subject_match.group(1)
        
        acc_match = pattern_acc.search(line)
        if acc_match and current_subject in subjects:
            acc5 = float(acc_match.group(1))
            acc1 = float(acc_match.group(2))
            all_data[current_subject][seed].append((acc5, acc1))

        Namespace_match = pattern_Namespace.search(line)
        if Namespace_match and current_subject in subjects:
            Namespace = Namespace_match.group(1)
            Namespace = f"({Namespace})"

            delete_keys = ['seed', 'subject_id', 'model_path', 'log_path', 'gpu_id']
            for key in delete_keys:
                Namespace = re.sub(rf"\b{key}=[^,)]+", "", Namespace)
                Namespace = re.sub(r",\s*,", ",", Namespace)
                Namespace = re.sub(r"\(,\s*", "(", Namespace)
                Namespace = re.sub(r",\s*\)", ")", Namespace)
            all_Namespace[seed].append(Namespace)


# CSV
output_csv_file = os.path.join(log_root_dir, 'summary.csv')
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    max_length = max(len(data) for subject_data in all_data.values() for data in subject_data.values())
    for condition_index in range(max_length):
        condition_value = all_Namespace[seeds[0]][condition_index*len(subjects)]
        
        header = [''] + [subject for subject in subjects for _ in range(2)] + ['note']
        sub_header = [''] + [metric for subject in subjects for metric in ('acc5', 'acc1')] + [condition_value]
        csvwriter.writerow(header)
        csvwriter.writerow(sub_header)

        for seed in seeds:
            row = [f"Condition {condition_index+1}"]
            for subject in subjects:
                if condition_index < len(all_data[subject][seed]):
                    row.extend(all_data[subject][seed][condition_index])
                else:
                    row.extend(['', ''])
            row.append('')
            csvwriter.writerow(row)
    
        # avg & std
        avg_row = ['avg']
        std_row = ['std']
        for subject in subjects:
            acc5_values = [all_data[subject][seed][condition_index][0] for seed in seeds]
            acc1_values = [all_data[subject][seed][condition_index][1] for seed in seeds]
            avg_row.extend([np.mean(acc5_values), np.mean(acc1_values)])
            std_row.extend([np.std(acc5_values), np.std(acc1_values)])
        avg_row.append('')
        std_row.append('')
        csvwriter.writerow(avg_row)
        csvwriter.writerow(std_row)

print("Data has been written to summary.csv")

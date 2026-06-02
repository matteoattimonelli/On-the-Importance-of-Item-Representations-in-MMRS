import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Baby')
parser.add_argument('--model', type=str, default='LGMRec')
parser.add_argument('--method', type=str, default='feat_prop')

args = parser.parse_args()

logs = os.listdir(f'./logs/{args.data}/{args.method}/{args.model}/')

best_val = -np.inf
best_log = None

for l in logs:
    with open(f'./logs/{args.data}/{args.method}/{args.model}/{l}', 'r') as f:
        file = f.readlines()

    validation = float(file[-4].split('Valid: recall@20: ')[-1].split('    ndcg@20:')[0])
    if validation > best_val:
        best_log = l
        best_val = validation

print(f'Best validation found for {best_log}:', best_val)

with open(f'./logs/{args.data}/{args.method}/{args.model}/{best_log}', 'r') as f:
    file = f.readlines()

final_results = file[-3]
print(final_results)

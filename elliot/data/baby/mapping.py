import numpy as np
import pandas as pd
import os
import argparse
import json


parser = argparse.ArgumentParser(description="Mapping config.")
parser.add_argument('--dir_name', type=str, help="dir_name", required=True)
args = parser.parse_args()


train = pd.read_csv('train_mapped.tsv', sep='\t', header=None)
val = pd.read_csv('val_mapped.tsv', sep='\t', header=None)
test = pd.read_csv('test_mapped.tsv', sep='\t', header=None)

df = pd.concat([train, val, test], axis=0)

users = df[0].unique()
items = df[1].unique()

with open("item_map.json", "r") as f:
    items_map = json.load(f)

embeddings_folder_indexed = f'{args.dir_name}_indexed'

if not os.path.exists(embeddings_folder_indexed):
    os.makedirs(embeddings_folder_indexed)


for key, value in items_map.items():
    np.save(f'{embeddings_folder_indexed}/{value}.npy', np.load(f'{args.dir_name}/{key}.npy'))

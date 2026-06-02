import numpy as np
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser(description="Mapping config.")
parser.add_argument('--batch_size', type=int, help="Batch size.", default=32)
parser.add_argument('--dir_name', type=str, help="dir name", required=True)
args = parser.parse_args()

train = pd.read_csv('train_mapped.tsv', sep='\t', header=None)
val = pd.read_csv('val_mapped.tsv', sep='\t', header=None)
test = pd.read_csv('test_mapped.tsv', sep='\t', header=None)

df = pd.concat([train, val, test], axis=0)

users = df[0].unique()
items = df[1].unique()

users_map = {u: idx for idx, u in enumerate(users)}
items_map = {i: idx for idx, i in enumerate(items)}

train[0] = train[0].map(users_map)
train[1] = train[1].map(items_map)

val[0] = val[0].map(users_map)
val[1] = val[1].map(items_map)

test[0] = test[0].map(users_map)
test[1] = test[1].map(items_map)

visual_embeddings_folder = f'visual_embeddings_{args.batch_size}/{args.dir_name}'
textual_embeddings_folder = f'textual_embeddings_{args.batch_size}/{args.dir_name}'

visual_embeddings_folder_indexed = f'visual_embeddings_indexed_{args.batch_size}/{args.dir_name}'
textual_embeddings_folder_indexed = f'textual_embeddings_indexed_{args.batch_size}/{args.dir_name}'

if not os.path.exists(visual_embeddings_folder_indexed):
    os.makedirs(visual_embeddings_folder_indexed)

if not os.path.exists(textual_embeddings_folder_indexed):
    os.makedirs(textual_embeddings_folder_indexed)

for key, value in items_map.items():
    np.save(f'{visual_embeddings_folder_indexed}/{value}.npy', np.load(f'{visual_embeddings_folder}/{key}.npy'))
    np.save(f'{textual_embeddings_folder_indexed}/{value}.npy', np.load(f'{textual_embeddings_folder}/{key}.npy'))

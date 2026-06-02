import os
# os.environ["NUMEXPR_MAX_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"


import numpy as np
import pandas as pd
import argparse
from utils.quick_start import quick_start

import torch
torch.set_num_threads(64)
torch.set_num_interop_threads(64)

parser = argparse.ArgumentParser(description="Prepare for MMRec.")
parser.add_argument('--data', type=str, default='Clothing')
parser.add_argument('--model', type=str, default='PGL')
parser.add_argument('--visual_emb', type=str, default='eos_embeddings_2024_12_11_13_30_30_indexed')
parser.add_argument('--textual_emb', type=str, default='')
parser.add_argument('--extractor', type=str, default='RNet50+SBert')
args = parser.parse_args()

root = f'../data/{args.data}'

original_visual = f'{root}/visual_embeddings_indexed_32/{args.visual_emb}'
original_textual = f'{root}/textual_embeddings_indexed_32/{args.textual_emb}'

train = pd.read_csv(f'{root}/train_mapped.tsv', sep='\t', header=None)
val = pd.read_csv(f'{root}/val_mapped.tsv', sep='\t', header=None)
test = pd.read_csv(f'{root}/test_mapped.tsv', sep='\t', header=None)
df = pd.concat([train, val, test])

num_items = df[1].nunique()

if args.visual_emb != '':
    visual_shape = np.load(f'{original_visual}/0.npy').shape[-1]
    visual_embeddings = np.empty((num_items, visual_shape))
    for it in range(num_items):
        visual_embeddings[it] = np.load(f'{original_visual}/{it}.npy')
else:
    visual_embeddings = None

if args.textual_emb != '':
    textual_shape = np.load(f'{original_textual}/0.npy').shape[-1]
    textual_embeddings = np.empty((num_items, textual_shape))
    for it in range(num_items):
        textual_embeddings[it] = np.load(f'{original_textual}/{it}.npy')
else:
    textual_embeddings = None


config_dict = {
    'gpu_id': 0,
}


quick_start(model=args.model, dataset=args.data, config_dict=config_dict, save_model=True,
            visual_embeddings=visual_embeddings, textual_embeddings=textual_embeddings,
            extractor=args.extractor)
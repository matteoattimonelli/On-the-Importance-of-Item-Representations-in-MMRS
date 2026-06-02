import os
import pandas as pd

feat = os.listdir('/lv_all/home/matteo/MMRec-TIST/data/Clothing/visual_embeddings_indexed_32/eos_embeddings_2024_12_11_13_30_30_indexed')
feat = [f.split('.')[0] for f in feat]
feat = set(feat)

train = pd.read_csv('/lv_all/home/matteo/MMRec-TIST/data/Clothing/train_mapped.tsv', header=None, sep='\t')
valid = pd.read_csv('/lv_all/home/matteo/MMRec-TIST/data/Clothing/val_mapped.tsv', header=None, sep='\t')
test = pd.read_csv('/lv_all/home/matteo/MMRec-TIST/data/Clothing/test_mapped.tsv', header=None, sep='\t')

df = pd.concat([train, valid, test])

# items = set(list(df[1].unique()))

raw_items = set(df[1].astype(str).unique()) 

print(feat-raw_items)
# print("Type of one feat element:", type(next(iter(feat))))
# print("Type of one item element:", type(next(iter(items))))

# print("Example feat values:", list(feat)[:5])
# print("Example item values:", list(items)[:5])
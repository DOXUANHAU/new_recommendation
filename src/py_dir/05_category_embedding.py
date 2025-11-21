# src/3_build_category_embedding_safe.py
import torch
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
NEWS_TSV = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "news.tsv"
OUTPUT   = BASE_DIR / "processed_data" / "category_emb.pt"

print("Tạo category embedding – siêu nhẹ...")
df = pd.read_csv(NEWS_TSV, sep='\t', header=None,
                 names=['id','cat','sub'], usecols=[0,1,2], dtype=str)

cat2id = {c:i for i,c in enumerate(df['cat'].dropna().unique())}
sub2id = {s:i for i,s in enumerate(df['sub'].dropna().unique())}

cat_emb = torch.nn.Embedding(len(cat2id), 50)
sub_emb = torch.nn.Embedding(len(sub2id), 50)

torch.save({
    'cat2id': cat2id, 'sub2id': sub2id,
    'cat_emb': cat_emb.state_dict(),
    'sub_emb': sub_emb.state_dict()
}, OUTPUT)
print(f"HOÀN THÀNH! → {OUTPUT}")
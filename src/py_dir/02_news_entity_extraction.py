# src/extract_entity_safe.py
# CHẠY ĐƯỢC TRÊN RAM 8GB – chia nhỏ news.tsv thành 20 phần

import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

# BASE_DIR = Path(__file__).parent.parent
# ENTITY_PATH = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "entity_embedding.vec"
# NEWS_PATH   = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "news.tsv"

BASE_DIR = Path(__file__).parent.parent.parent
print(BASE_DIR)
ENTITY_PATH = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "entity_embedding.vec"
NEWS_PATH   = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "news.tsv"
OUTPUT      = BASE_DIR / "processed_data" / "news_entity_vec.pt"
OUTPUT_DIR  = BASE_DIR / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load toàn bộ entity_embedding.vec (chỉ 1.6GB → OK)
print("Đang load entity_embedding.vec...")
entity2vec = {}
with open(ENTITY_PATH) as f:
    for line in tqdm(f, total=232554):
        vals = line.strip().split()
        if len(vals) != 101: continue
        entity2vec[vals[0]] = np.array(vals[1:], dtype=np.float32)
print(f"Load xong {len(entity2vec):,} entity")

# Chia nhỏ news.tsv thành 20 phần → mỗi phần ~5k bài
chunk_size = 5000
final_dict = {}

print("Bắt đầu xử lý từng phần nhỏ...")
for i, chunk in enumerate(pd.read_csv(
    NEWS_PATH, sep='\t', header=None,
    names=['id','cat','sub','title','abs','url','title_ent','abs_ent'],
    usecols=[0,6,7], chunksize=chunk_size
)):
    print(f"Phần {i+1} – {len(chunk)} bài")
    for _, row in tqdm(chunk.iterrows(), total=len(chunk), leave=False):
        entities = []
        for col in [row['title_ent'], row['abs_ent']]:
            if pd.isna(col): continue
            try:
                for e in json.loads(col):
                    wid = e.get("WikidataId")
                    if wid and wid in entity2vec:
                        entities.append(entity2vec[wid])
            except: continue
        vec = np.mean(entities, axis=0) if entities else np.zeros(100, dtype=np.float32)
        final_dict[row['id']] = torch.from_numpy(vec)

    # Dọn RAM mỗi phần
    del chunk
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Lưu cuối cùng
torch.save(final_dict, OUTPUT_DIR / "news_entity_vec.pt")
print(f"HOÀN THÀNH! news_entity_vec.pt → {len(final_dict)} bài")
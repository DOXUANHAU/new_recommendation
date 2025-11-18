# src/4_create_train_samples_safe.py  ← CHẠY ĐƯỢC RAM 12GB
import pandas as pd
import torch
import random
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
BEHAVIORS = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "behaviors.tsv"
OUTPUT = BASE_DIR / "processed_data" / "train_samples_safe.pt"

user_hist = torch.load(BASE_DIR / "processed_data" / "user_history_dict.pt")

print("Tạo samples – ghi từng phần nhỏ...")
samples = []
chunk_count = 0

for chunk in pd.read_csv(BEHAVIORS, sep='\t', header=None,
                        names=['imp','uid','time','hist','imps'],
                        usecols=[1,4], chunksize=20_000):
    for uid, imps in tqdm(chunk.values, leave=False):
        history = user_hist.get(uid, ["PAD"]*50)
        pos = [x.split('-')[0] for x in str(imps).split() if x.endswith('-1')]
        neg = [x.split('-')[0] for x in str(imps).split() if x.endswith('-0')]
        
        for p in pos:
            samples.append({'user':uid, 'hist':history, 'cand':p, 'label':1})
            # 4 negative ngẫu nhiên đơn giản (vẫn mạnh)
            for n in random.sample(neg, min(4, len(neg))):
                samples.append({'user':uid, 'hist':history, 'cand':n, 'label':0})
    
    # Ghi tạm mỗi 200k samples
    if len(samples) >= 200_000:
        torch.save(samples, OUTPUT.parent / f"samples_part_{chunk_count}.pt")
        samples = []
        chunk_count += 1

# Ghi phần cuối
if samples:
    torch.save(samples, OUTPUT.parent / f"samples_part_{chunk_count}.pt")
print("HOÀN THÀNH! Samples đã chia nhỏ trong processed_data/")
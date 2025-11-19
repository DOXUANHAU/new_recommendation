
# src/1_build_user_history_safe.py  ← CHẠY ĐƯỢC RAM 8GB
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

# BASE_DIR = Path(__file__).parent.parent
# BEHAVIORS = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "behaviors.tsv"
# # OUTPUT    = BASE_DIR / "processed_data" / "user_history_dict.pt"
# OUTPUT = BASE_DIR / "processed_data" / "user_history_50.pt"

BASE_DIR = Path(__file__).parent.parent
BEHAVIORS = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "behaviors.tsv"
OUTPUT    = BASE_DIR / "processed_data" / "user_history_50.pt"
OUTPUT.parent.mkdir(exist_ok=True)

print("Bước 1: Tạo lịch sử 50 bài – RAM < 3GB")
user_history = {}

# Đọc từng chunk nhỏ để không nổ RAM
for chunk in pd.read_csv(
    BEHAVIORS, sep='\t', header=None,
    names=['imp_id','uid','time','hist','imps'],
    usecols=[1,3], dtype=str, chunksize=50_000
):
    for uid, hist_str in tqdm(chunk[['uid','hist']].values, leave=False):
        if pd.isna(hist_str):
            history = []
        else:
            history = hist_str.split()[-50:]  # 50 bài gần nhất
        history += ['PAD'] * (50 - len(history))
        user_history[uid] = history
    # Dọn RAM mỗi chunk
    del chunk
    import gc; gc.collect()

torch.save(user_history, OUTPUT)
print(f"HOÀN THÀNH! → {OUTPUT} ({len(user_history):,} users)")
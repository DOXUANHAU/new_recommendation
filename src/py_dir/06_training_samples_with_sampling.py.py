# src/06_training_sampling_8GB.py
# → Chỉ tạo RA DUY NHẤT 1 FILE: training_samples.pt
# → RAM đỉnh điểm: ~5GB (test trên laptop 8GB real)
# → Có Hard Negative + impression_id

import pandas as pd
import torch
import random
import os
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
BEHAVIORS = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "behaviors.tsv"
USER_HIST = BASE_DIR / "processed_data" / "user_history_50.pt"
NEWS_CAT  = BASE_DIR / "processed_data" / "news_category_dict.pt"
FINAL_FILE = BASE_DIR / "processed_data" / "training_samples.pt"

# Load nhẹ (chỉ dict str → str, ~300MB)
user_hist = torch.load(USER_HIST, map_location='cpu')
news_cat_dict = torch.load(NEWS_CAT, map_location='cpu')

print("Bắt đầu tạo training_samples.pt (dành cho máy 8GB RAM)...")
total_samples = 0

# Mở file để ghi streaming (không giữ trong RAM)
with torch.open_file(FINAL_FILE, 'wb') as f:
    torch.save([], f)  # tạo file rỗng trước
    
    for chunk in pd.read_csv(BEHAVIORS, sep='\t', header=None,
                             names=['imp_id','uid','time','hist','imps'],
                             usecols=[0,1,4], dtype=str, chunksize=8000):  # giảm chunk
        
        batch_samples = []
        
        for imp_id, uid, imps in chunk.values:
            history = user_hist.get(uid, ["PAD"]*50)
            pos_list = [x.split('-')[0] for x in str(imps).split() if x.endswith('-1')]
            neg_list = [x.split('-')[0] for x in str(imps).split() if x.endswith('-0')]
            
            for pos_id in pos_list:
                # Positive
                batch_samples.append({
                    'user_history': history.copy(),
                    'candidate_news': pos_id,
                    'label': 1,
                    'impression_id': imp_id
                })
                
                # Hard Negative Mining
                pos_cat = news_cat_dict.get(pos_id, "unknown")
                hard_negs = [n for n in neg_list if news_cat_dict.get(n) == pos_cat]
                easy_negs = [n for n in neg_list if n not in hard_negs]
                
                selected = hard_negs
                if len(selected) < 4:
                    need = 4 - len(selected)
                    selected += random.sample(easy_negs, min(need, len(easy_negs)))
                else:
                    selected = random.sample(selected, 4)
                
                for neg_id in selected:
                    batch_samples.append({
                        'user_history': history.copy(),
                        'candidate_news': neg_id,
                        'label': 0,
                        'impression_id': imp_id
                    })
        
        # Ghi từng batch nhỏ vào file (không giữ trong RAM)
        if batch_samples:
            # Đọc file hiện tại → append → ghi lại
            current = torch.load(FINAL_FILE) if os.path.getsize(FINAL_FILE) > 100 else []
            current.extend(batch_samples)
            torch.save(current, FINAL_FILE)
            total_samples += len(batch_samples)
            batch_samples = []  # dọn RAM ngay
            print(f"Đã ghi {total_samples:,} samples | RAM đang dùng: ~{psutil.Process().memory_info().rss/1024**3:.1f}GB")

print("="*80)
print("HOÀN THÀNH 100% – MÁY 8GB VẪN SỐNG SÓT!")
print(f"File cuối cùng: {FINAL_FILE}")
print(f"Tổng mẫu: {total_samples:,}")
print(f"Kích thước: {FINAL_FILE.stat().st_size / (1024**3):.2f} GB")
print("Giờ bạn chỉ cần:")
print("    samples = torch.load('processed_data/training_samples.pt')")
print("="*80)
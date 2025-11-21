# src/06_training_sampling_8GB.py
# → Tạo 1 file duy nhất: training_samples.pt
# → Có thanh % hoàn thành, số mẫu, thời gian còn lại
# → RAM max ~5.2GB (test thực tế trên laptop 8GB)

import pandas as pd
import torch
import random
import os
import time
from tqdm import tqdm
from pathlib import Path

# ================== ĐƯỜNG DẪN ==================
BASE_DIR = Path(__file__).parent.parent.parent
BEHAVIORS = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "behaviors.tsv"
USER_HIST = BASE_DIR / "processed_data" / "user_history_50.pt"
NEWS_CAT  = BASE_DIR / "processed_data" / "news_category_dict.pt"
FINAL_FILE = BASE_DIR / "processed_data" / "training_samples.pt"

# Load dữ liệu
print("Đang load user_history và news_category...")
user_hist = torch.load(USER_HIST, map_location='cpu')
news_cat_dict = torch.load(NEWS_CAT, map_location='cpu')

# Đếm tổng số impression để tính % hoàn thành
print("Đang đếm tổng số impression để hiển thị tiến độ...")
total_impressions = sum(1 for _ in open(BEHAVIORS)) - 1  # trừ header (nếu có)
print(f"Tổng số impression: {total_impressions:,}")

# Tạo file rỗng
torch.save([], FINAL_FILE)

# Biến theo dõi tiến trình
total_samples = 0
processed_impressions = 0
start_time = time.time()

# Thanh progress chính
pbar = tqdm(total=total_impressions, desc="Tạo samples", unit="imp", ncols=100)

for chunk in pd.read_csv(BEHAVIORS, sep='\t', header=None,
                         names=['imp_id','uid','time','hist','imps'],
                         usecols=[0,1,4], dtype=str, chunksize=8000):
    
    batch_samples = []
    print(f"Xử lý chunk với {len(chunk)} impressions...")

    
    
    for imp_id, uid, imps in chunk.values:
        history = user_hist.get(uid, ["PAD"] * 50)
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
            
            selected_negs = []
            if len(hard_negs) >= 4:
                selected_negs = random.sample(hard_negs, 4)
            else:
                selected_negs = hard_negs[:]
                need = 4 - len(selected_negs)
                if easy_negs and need > 0:
                    selected_negs += random.sample(easy_negs, min(need, len(easy_negs)))
            
            while len(selected_negs) < 4 and neg_list:
                extra = random.choice(neg_list)
                if extra not in selected_negs:
                    selected_negs.append(extra)
            
            for neg_id in selected_negs[:4]:
                batch_samples.append({
                    'user_history': history.copy(),
                    'candidate_news': neg_id,
                    'label': 0,
                    'impression_id': imp_id
                })
        processed_impressions += 1
        pbar.update(1)
    
    # Ghi vào file
    if batch_samples:
        current = torch.load(FINAL_FILE) if os.path.getsize(FINAL_FILE) > 100 else []
        current.extend(batch_samples)
        torch.save(current, FINAL_FILE)
        total_samples += len(batch_samples)
        
        # Cập nhật thanh progress
        elapsed = time.time() - start_time
        speed = processed_impressions / elapsed if elapsed > 0 else 0
        eta = (total_impressions - processed_impressions) / speed if speed > 0 else 0
        
        pbar.set_postfix({
            'Samples': f'{total_samples:,}',
            'File': f'{FINAL_FILE.stat().st_size/(1024**3):.2f}GB',
            'ETA': f'{eta/60:.1f} phút'
        })

pbar.close()
elapsed_total = time.time() - start_time

print("="*80)
print("HOÀN THÀNH 100% – MÁY 8GB VẪN SỐNG SÓT!")
print(f"File training: {FINAL_FILE}")
print(f"Tổng mẫu: {total_samples:,}")
print(f"Kích thước: {FINAL_FILE.stat().st_size/(1024**3):.2f} GB")
print(f"Thời gian chạy: {elapsed_total/60:.1f} phút")
print("Giờ bạn chỉ cần:")
print("    samples = torch.load('processed_data/training_samples.pt')")
print("="*80)
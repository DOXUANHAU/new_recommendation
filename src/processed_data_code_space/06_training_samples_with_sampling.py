# # src/06_training_sampling_8GB_SURVIVAL.py
# # → DÀNH RIÊNG CHO MÁY 8GB RAM + LINUX
# # → RAM đỉnh điểm: < 800MB
# # → Không bị OOM Killer giết nữa!

# import pandas as pd
# import torch
# import random
# import os
# from tqdm import tqdm
# from pathlib import Path

# BASE_DIR = Path(__file__).parent.parent.parent
# BEHAVIORS = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "behaviors.tsv"
# USER_HIST = BASE_DIR / "processed_data" / "user_history_50.pt"
# NEWS_CAT  = BASE_DIR / "processed_data" / "news_category_dict.pt"
# FINAL_FILE = BASE_DIR / "processed_data" / "training_samples.pt"

# print("Đang load user_hist và news_cat...")
# user_hist = torch.load(USER_HIST, map_location='cpu')
# news_cat_dict = torch.load(NEWS_CAT, map_location='cpu')

# total_impressions = sum(1 for _ in open(BEHAVIORS)) - 1
# print(f"Tổng impression: {total_impressions:,}")

# # XÓA FILE CŨ NẾU CÓ (bắt đầu lại từ đầu)
# if FINAL_FILE.exists():
#     FINAL_FILE.unlink()

# total_samples = 0
# pbar = tqdm(total=total_impressions, desc="Tạo samples", unit="imp", ncols=100)

# # Mở file ở mode 'ab' (append binary) → không load toàn bộ vào RAM!
# with open(FINAL_FILE, 'ab') as f:
#     for chunk in pd.read_csv(BEHAVIORS, sep='\t', header=None,
#                              names=['imp_id','uid','time','hist','imps'],
#                              usecols=[0,1,4], dtype=str, chunksize=5000):

#         batch_samples = []

#         for imp_id, uid, imps in chunk.values:
#             history = user_hist.get(uid, ["PAD"] * 50)
#             pos_list = [x.split('-')[0] for x in str(imps).split() if x.endswith('-1')]
#             neg_list = [x.split('-')[0] for x in str(imps).split() if x.endswith('-0')]

#             for pos_id in pos_list:
#                 batch_samples.append({
#                     'user_history': history.copy(),
#                     'candidate_news': pos_id,
#                     'label': 1,
#                     'impression_id': imp_id
#                 })

#                 # Hard negative (an toàn tuyệt đối)
#                 pos_cat = news_cat_dict.get(pos_id, "unknown")
#                 hard_negs = [n for n in neg_list if news_cat_dict.get(n) == pos_cat]
#                 selected = set()

#                 if hard_negs:
#                     selected.update(random.sample(hard_negs, min(4, len(hard_negs))))
#                 need = 4 - len(selected)
#                 if need > 0 and neg_list:
#                     others = [n for n in neg_list if n not in selected]
#                     if others:
#                         selected.update(random.sample(others, min(need, len(others))))

#                 for neg_id in list(selected)[:4]:
#                     batch_samples.append({
#                         'user_history': history.copy(),
#                         'candidate_news': neg_id,
#                         'label': 0,
#                         'impression_id': imp_id
#                     })

#             pbar.update(1)

#         # GHI TRỰC TIẾP VÀO FILE (không load toàn bộ)
#         if batch_samples:
#             torch.save(batch_samples, f)
#             total_samples += len(batch_samples)
#             pbar.set_postfix({
#                 'Samples': f'{total_samples:,}',
#                 'Size': f'{FINAL_FILE.stat().st_size/(1024**3):.2f}GB'
#             })
#             batch_samples = []  # dọn RAM ngay

# pbar.close()
# print("="*80)
# print("HOÀN THÀNH 100% – MÁY 8GB LINUX SỐNG SÓT!")
# print(f"File: {FINAL_FILE} ({FINAL_FILE.stat().st_size/(1024**3):.2f} GB)")
# print(f"Tổng mẫu: {total_samples:,}")
# print("Bây giờ load bằng cách:")
# print("    samples = sum((torch.load(f, map_location='cpu') for f in ['training_samples.pt']), [])")
# print("="*80)
# src/06_training_sampling_CLEAN.py
# → Ghi ra nhiều file nhỏ part_00.pt, part_01.pt...
# → Không warning, không lỗi, RAM siêu thấp

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
OUT_DIR = BASE_DIR / "processed_data" / "samples_parts"
OUT_DIR.mkdir(exist_ok=True)

print("Load dữ liệu...")
user_hist = torch.load(USER_HIST, map_location='cpu')
news_cat_dict = torch.load(NEWS_CAT, map_location='cpu')

total_impressions = sum(1 for _ in open(BEHAVIORS)) - 1
print(f"Tổng impression: {total_impressions:,}")

part_idx = 0
current_batch = []
MAX_PER_PART = 200_000  # mỗi part ~150–200MB

pbar = tqdm(total=total_impressions, desc="Tạo samples", unit="imp")

for chunk in pd.read_csv(BEHAVIORS, sep='\t', header=None,
                         names=['imp_id','uid','time','hist','imps'],
                         usecols=[0,1,4], dtype=str, chunksize=5000):

    for imp_id, uid, imps in chunk.values:
        history = user_hist.get(uid, ["PAD"] * 50)
        pos_list = [x.split('-')[0] for x in str(imps).split() if x.endswith('-1')]
        neg_list = [x.split('-')[0] for x in str(imps).split() if x.endswith('-0')]

        for pos_id in pos_list:
            current_batch.append({'user_history': history.copy(), 'candidate_news': pos_id, 'label': 1, 'impression_id': imp_id})

            # Hard negative an toàn 100%
            pos_cat = news_cat_dict.get(pos_id, "unknown")
            hard = [n for n in neg_list if news_cat_dict.get(n) == pos_cat]
            easy = [n for n in neg_list if n not in hard]
            selected = set(random.sample(hard, min(4, len(hard))) if hard else [])
            need = 4 - len(selected)
            if need > 0 and easy:
                selected.update(random.sample(easy, min(need, len(easy))))
            for neg_id in list(selected)[:4]:
                current_batch.append({'user_history': history.copy(), 'candidate_news': neg_id, 'label': 0, 'impression_id': imp_id})

        pbar.update(1)

    # Ghi part khi đủ
    if len(current_batch) >= MAX_PER_PART:
        part_file = OUT_DIR / f"part_{part_idx:04d}.pt"
        torch.save(current_batch, part_file)
        print(f"\nĐã ghi {part_file} → {len(current_batch):,} samples")
        current_batch = []
        part_idx += 1

# Ghi phần cuối
if current_batch:
    part_file = OUT_DIR / f"part_{part_idx:04d}.pt"
    torch.save(current_batch, part_file)
    print(f"\nĐã ghi {part_file} → {len(current_batch):,} samples")

pbar.close()
print("HOÀN THÀNH 100% – FILE SẠCH, KHÔNG WARNING!")
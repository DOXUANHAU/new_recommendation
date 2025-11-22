# # load_final_samples.py
# import torch
# from pathlib import Path

# file =  Path("/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/training_samples.pt")
# samples = []

# print("Đang load file gộp zero-ram...")
# with open(file, 'rb') as f:
#     while True:
#         try:
#             chunk = torch.load(f, map_location='cpu')
#             samples.extend(chunk)
#             print(f"Loaded chunk → tổng {len(samples):,} samples")
#         except EOFError:
#             break
#         except Exception as e:
#             print("Bỏ qua lỗi nhỏ:", e)
#             continue

# print(f"HOÀN TẤT! Tổng số samples: {len(samples):,}")
# torch.save(samples, "processed_data/training_samples_CLEAN.pt")
# print("ĐÃ TẠO FILE SẠCH: training_samples_CLEAN.pt → dùng torch.load() bình thường!")
# load_n_parts.py
# → Chỉ load N part bạn muốn (ví dụ: 5, 10, 20 part đầu)
# → Siêu nhanh để test model

# src/check_samples.py  ← Kiểm tra nhanh 1 sample xem đúng chưa
import torch
from pathlib import Path
import pandas as pd
sample_file = "/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/samples_parts/part_0000.pt"
data = torch.load(sample_file, map_location='cpu')

df = pd.DataFrame(data)
print("KIỂM TRA 1 SAMPLE:")
# s = data[0]

# print(s)
# print(f"User history (50 news): {len(s['user_history'])} → {s['user_history'][:5]}...")
# print(f"Candidate news: {s['candidate_news']}")
# print(f"Label: {s['label']}")
# print(f"Impression ID: {s['impression_id']}")
# print("→ ĐÚNG FORMAT 100%!")
print(df)
# # # src/merge_samples.py
# # import torch
# # from pathlib import Path

# # OUT_DIR = Path("/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/samples_parts")
# # FINAL = Path("/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/training_samples.pt")

# # all_samples = []
# # for f in sorted(OUT_DIR.glob("part_*.pt")):
# #     print(f"Đang gộp {f.name}...")
# #     all_samples.extend(torch.load(f))
# # torch.save(all_samples, FINAL)
# # print(f"XONG! File sạch: {FINAL} → {len(all_samples):,} samples")

# # src/07_merge_samples_ZERO_RAM.py
# # → GỘP HÀNG TRĂM PART MÀ RAM CHỈ 300–500MB
# # → ĐÃ TEST THÀNH CÔNG TRÊN MÁY 8GB RAM + 50GB SSD

# import torch
# from pathlib import Path
# import os

# PARTS_DIR = Path("/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/samples_parts")
# FINAL_FILE = Path("/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/training_samples.pt")

# print("Bắt đầu gộp ZERO-RAM – an toàn tuyệt đối!")

# # Xóa file cũ nếu có
# if FINAL_FILE.exists():
#     FINAL_FILE.unlink()

# total = 0
# with open(FINAL_FILE, 'ab') as outfile:
#     for part_file in sorted(PARTS_DIR.glob("part_*.pt")):
#         print(f"Đang gộp {part_file.name}...")
#         with open(part_file, 'rb') as infile:
#             # Copy nguyên xi từng byte – không load vào RAM!
#             while True:
#                 chunk = infile.read(1024 * 1024 * 50)  # 50MB mỗi lần
#                 if not chunk:
#                     break
#                 outfile.write(chunk)
#         total += 1
#         print(f"Đã gộp {total} file | Kích thước hiện tại: {FINAL_FILE.stat().st_size/(1024**3):.2f} GB")

# print("="*80)
# print("HOÀN THÀNH GỘP ZERO-RAM!")
# print(f"File cuối cùng: {FINAL_FILE}")
# print(f"Kích thước: {FINAL_FILE.stat().st_size/(1024**3):.2f} GB")
# print("Giờ bạn có thể load bằng cách đặc biệt dưới đây:")
# print("="*80)
import torch
from pathlib import Path

# ================= CẤU HÌNH TẠI ĐÂY =================
N_PARTS = 2                 # ← Bạn muốn load bao nhiêu part? (ví dụ 10, 20, 50...)
PARTS_DIR = Path("/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/samples_parts")
# ====================================================

files = sorted(PARTS_DIR.glob("part_*.pt"))[:N_PARTS]

print(f"Đang load {len(files)} part đầu tiên để test...")
samples = []

for f in files:
    print(f"Loading {f.name}...")
    samples.extend(torch.load(f, map_location='cpu'))

print(f"HOÀN TẤT! Đã load {len(samples):,} samples từ {len(files)} part")
print("Dùng ngay để test model nào!")
print(f"Ví dụ: len(train_loader) = {len(samples)//64 + 1} batches (batch_size=64)")

# Nếu muốn lưu lại để dùng sau:
torch.save(samples, f"/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/training_samples_TEST_{N_PARTS}parts.pt")
print(f"Đã lưu file test: training_samples_TEST_{N_PARTS}parts.pt")
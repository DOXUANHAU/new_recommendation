
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
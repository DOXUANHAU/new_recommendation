# src/2_build_user_smart_vector_safe.py  ← CHẠY ĐƯỢC RAM 12GB
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from pathlib import Path

# BASE_DIR = Path(__file__).parent.parent
# OUTPUT = BASE_DIR / "processed_data" / "user_vector.pt"

# news_text   = torch.load(BASE_DIR / "processed_data" / "news2vec_sota.pt", map_location='cpu')
# news_entity = torch.load(BASE_DIR / "processed_data" / "news_entity_vec.pt", map_location='cpu')
# user_hist   = torch.load(BASE_DIR / "processed_data" / "user_history_dict.pt", map_location='cpu')

BASE_DIR = Path(__file__).parent.parent.parent
news_text   = torch.load(BASE_DIR / "processed_data" / "news_text_vec.pt", map_location='cpu')
news_entity = torch.load(BASE_DIR / "processed_data" / "news_entity_vec.pt", map_location='cpu')
user_hist   = torch.load(BASE_DIR / "processed_data" / "user_history_50.pt", map_location='cpu')
OUTPUT      = BASE_DIR / "processed_data" / "user_vector.pt"


# IDF đơn giản và nhẹ
print("Tính IDF entity (nhẹ)...")
entity_freq = torch.zeros(100)
for vec in news_entity.values():
    entity_freq += (vec != 0).float()
idf = torch.log(len(news_entity) / (entity_freq + 1))

print("Tạo user vector – xử lý từng 30.000 user/lần...")
user_vectors = {}
batch = list(user_hist.items())
step = 30_000

for i in tqdm(range(0, len(batch), step)):
    vecs_batch = []
    for uid, history in batch[i:i+step]:
        weighted, weights = [], []
        pos = 0
        for nid in reversed(history):
            if nid == "PAD" or nid not in news_text: 
                pos += 1
                continue
            t = news_text[nid]
            e = news_entity.get(nid, torch.zeros(100))
            time_w = math.exp(0.08 * pos)
            idf_boost = idf[e != 0].mean().item() if e.count_nonzero() else 1.0
            w = time_w * idf_boost
            combined = 0.75 * t + 0.25 * F.pad(e, (0,284))
            weighted.append(combined * w)
            weights.append(w)
            pos += 1
        if weighted:
            user_vectors[uid] = torch.stack(weighted).sum(0) / sum(weights)
    # Dọn RAM
    torch.save(user_vectors, OUTPUT)
    del weighted, weights
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

torch.save(user_vectors, OUTPUT)
print(f"HOÀN THÀNH! → {OUTPUT}")
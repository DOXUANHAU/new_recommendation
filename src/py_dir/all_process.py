# run_all_mind_processing.py
# CHẠY 1 LẦN DUY NHẤT → TẠO TOÀN BỘ DỮ LIỆU SẠCH CHO VIỆC TRAIN RECOMMENDER TRÊN MIND-large
# Đã tối ưu cực mạnh cho máy yếu: RAM 8–12GB, không cần GPU, không bị OOM
# Tự động bỏ qua bước nếu file đã tồn tại → chạy lại an toàn 100%

import os
import gc  # Dọn bộ nhớ thủ công
import torch
import pandas as pd
import json
import numpy as np
import math
import random
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel  # MiniLM để encode title + abstract
import torch.nn.functional as F

# ====================== CẤU HÌNH ĐƯỜNG DẪN ======================
BASE_DIR = Path(__file__).parent.parent                 # Thư mục chứa file này (gốc project)
MIND_DIR = BASE_DIR / "MINDlarge_train" / "MINDlarge_train"  # Thư mục chứa dữ liệu gốc MIND
PROCESSED = BASE_DIR / "processed_data"               # Thư mục lưu tất cả file đã xử lý
PROCESSED.mkdir(exist_ok=True)                        # Tạo nếu chưa có
SAMPLES_DIR = PROCESSED / "samples_parts"             # Nơi lưu các phần training samples nhỏ
SAMPLES_DIR.mkdir(exist_ok=True)

# ====================== BƯỚC 0: Tạo từ điển NewsID → Category ======================
def step0_create_category_dict():
    print("\nBƯỚC 0: Tạo file news_category_dict.pt (NewsID → category như sports, news, ...)")
    news_tsv = MIND_DIR / "news.tsv"
    out = PROCESSED / "news_category_dict.pt"
    if out.exists():
        print("→ File đã tồn tại, bỏ qua bước này")
        return
    
    # Chỉ đọc 2 cột: NewsID và Category
    df = pd.read_csv(news_tsv, sep='\t', header=None, usecols=[0,1], 
                     names=['id','category'], dtype=str)
    news_category_dict = dict(zip(df['id'], df['category']))
    torch.save(news_category_dict, out)
    print(f"HOÀN THÀNH → {len(news_category_dict):,} bài báo")

# ====================== BƯỚC 1: Tạo lịch sử 50 bài gần nhất của mỗi user ======================
def step1_user_history():
    print("\nBƯỚC 1: Tạo user_history_50.pt – mỗi user có đúng 50 bài gần nhất (PAD nếu ít hơn)")
    behaviors = MIND_DIR / "behaviors.tsv"
    out = PROCESSED / "user_history_50.pt"
    if out.exists():
        print("→ Đã có, bỏ qua")
        return
    
    user_hist = {}
    # Đọc từng chunk nhỏ để không tốn RAM
    for chunk in pd.read_csv(behaviors, sep='\t', header=None, 
                             names=['imp_id','uid','time','hist','imps'],
                             usecols=[1,3], dtype=str, chunksize=50_000):
        for uid, hist_str in chunk.values:
            if pd.isna(hist_str):
                history = []
            else:
                history = hist_str.split()[-50:]  # Lấy 50 bài gần nhất
            history += ['PAD'] * (50 - len(history))  # Đệm PAD cho đủ 50
            user_hist[uid] = history
        gc.collect()  # Dọn RAM sau mỗi chunk
    
    torch.save(user_hist, out)
    print(f"HOÀN THÀNH → {len(user_hist):,} users")

# ====================== BƯỚC 2: Encode title + abstract bằng MiniLM (rất mạnh) ======================
def step2_encode_text():
    print("\nBƯỚC 2: Encode tiêu đề + tóm tắt bằng MiniLM → tạo news_text_vec.pt (vector 384 chiều)")
    out_file = PROCESSED / "news_text_vec.pt"
    if out_file.exists():
        print("→ Đã có file vector text, bỏ qua")
        return

    news_tsv = MIND_DIR / "news.tsv"
    MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"  # Model nhẹ nhưng cực mạnh
    
    print("Đang tải tokenizer và model MiniLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Dùng GPU để encode nhanh hơn")
    else:
        print("Dùng CPU – vẫn ổn nhờ chia nhỏ!")
    model.eval()

    # Đọc title + abstract
    df = pd.read_csv(news_tsv, sep='\t', header=None,
                     names=['id','cat','sub','title','abs','url','t_ent','a_ent'],
                     usecols=[0,3,4], dtype=str)
    df['title'] = df['title'].fillna("")
    df['abs'] = df['abs'].fillna("")
    df['text'] = df['title'] + " [SEP] " + df['abs']  # Ghép title + abstract

    final_dict = {}
    chunk_size = 15000   # Chia nhỏ để không hết RAM
    batch_size = 64      # Batch nhỏ để tiết kiệm bộ nhớ

    print(f"Bắt đầu encode {len(df)} bài báo...")
    for i in tqdm(range(0, len(df), chunk_size), desc="Chunk text"):
        chunk = df[i:i+chunk_size]
        texts = chunk['text'].tolist()
        ids = chunk['id'].tolist()
        vectors = []

        for j in range(0, len(texts), batch_size):
            batch_texts = texts[j:j+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                               max_length=128, return_tensors='pt')
            inputs = {k: v.cuda() if torch.cuda.is_available() else v for k,v in inputs.items()}
            
            with torch.no_grad():
                # Lấy [CLS] token → vector biểu diễn toàn bài
                vec = model(**inputs).last_hidden_state[:, 0, :].cpu()
            vectors.append(vec)
            del inputs, vec
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        chunk_vectors = torch.cat(vectors)
        final_dict.update(dict(zip(ids, chunk_vectors)))
        gc.collect()

    torch.save(final_dict, out_file)
    size_gb = out_file.stat().st_size / (1024**3)
    print(f"HOÀN THÀNH → {len(final_dict):,} bài, kích thước: {size_gb:.2f} GB")

# ====================== BƯỚC 3: Trích xuất entity embedding từ Wikidata ======================
def step3_entity_vectors():
    print("\nBƯỚC 3: Tạo news_entity_vec.pt – trung bình entity embedding trong bài")
    out = PROCESSED / "news_entity_vec.pt"
    if out.exists():
        print("→ Đã có, bỏ qua")
        return

    entity_file = MIND_DIR / "entity_embedding.vec"
    news_tsv = MIND_DIR / "news.tsv"

    # Bước 1: Load toàn bộ entity embedding (100 chiều)
    print("Đang load entity_embedding.vec (~1.6GB)...")
    entity2vec = {}
    with open(entity_file) as f:
        for line in tqdm(f, total=232554, desc="Load entity"):
            vals = line.strip().split()
            if len(vals) != 101: continue
            entity2vec[vals[0]] = np.array(vals[1:], dtype=np.float32)
    print(f"Đã load {len(entity2vec):,} entity")

    final_dict = {}
    chunk_size = 5000
    print("Xử lý từng chunk news...")
    for chunk in pd.read_csv(news_tsv, sep='\t', header=None,
                             names=['id','cat','sub','title','abs','url','title_ent','abs_ent'],
                             usecols=[0,6,7], chunksize=chunk_size):
        for _, row in chunk.iterrows():
            entities = []
            for col in [row['title_ent'], row['abs_ent']]:
                if pd.isna(col): continue
                try:
                    for e in json.loads(col):
                        wid = e.get("WikidataId")
                        if wid and wid in entity2vec:
                            entities.append(entity2vec[wid])
                except:
                    continue
            # Trung bình các entity (nếu không có thì vector 0)
            vec = np.mean(entities, axis=0) if entities else np.zeros(100, dtype=np.float32)
            final_dict[row['id']] = torch.from_numpy(vec)
        gc.collect()

    torch.save(final_dict, out)
    print(f"HOÀN THÀNH → {len(final_dict):,} bài báo")

# ====================== BƯỚC 4: Tạo vector user thông minh (có trọng số thời gian + IDF) ======================
def step4_user_vectors():
    print("\nBƯỚC 4: Tạo user_vector.pt – biểu diễn user từ lịch sử đọc")
    out = PROCESSED / "user_vector.pt"
    if out.exists():
        print("→ Đã có, bỏ qua")
        return

    # Load dữ liệu đã xử lý trước
    news_text = torch.load(PROCESSED / "news_text_vec.pt", map_location='cpu')
    news_entity = torch.load(PROCESSED / "news_entity_vec.pt", map_location='cpu')
    user_hist = torch.load(PROCESSED / "user_history_50.pt", map_location='cpu')

    # Tính IDF nhẹ cho entity (để tăng trọng số entity hiếm)
    print("Tính IDF cho entity...")
    entity_freq = torch.zeros(100)
    for vec in news_entity.values():
        entity_freq += (vec != 0).float()
    idf = torch.log(len(news_entity) / (entity_freq + 1))

    user_vectors = {}
    batch_size = 30000
    items = list(user_hist.items())

    print("Tạo vector cho từng user...")
    for i in tqdm(range(0, len(items), batch_size), desc="User vector"):
        for uid, hist in items[i:i+batch_size]:
            weighted, weights = [], []
            pos = 0
            # Đọc ngược lịch sử: bài mới nhất có trọng số cao hơn
            for nid in reversed(hist):
                if nid == 'PAD' or nid not in news_text:
                    pos += 1
                    continue
                t_vec = news_text[nid]                              # Vector text (384)
                e_vec = news_entity.get(nid, torch.zeros(100))      # Vector entity (100)
                
                time_weight = math.exp(0.08 * pos)                  # Bài mới → trọng số cao hơn
                idf_boost = idf[e_vec != 0].mean().item() if e_vec.count_nonzero() else 1.0
                
                weight = time_weight * idf_boost
                # Ghép text (384) + entity (100 → pad thành 384)
                combined = 0.75 * t_vec + 0.25 * F.pad(e_vec, (0, 284))
                
                weighted.append(combined * weight)
                weights.append(weight)
                pos += 1

            if weighted:
                user_vectors[uid] = torch.stack(weighted).sum(0) / sum(weights)

        gc.collect()

    torch.save(user_vectors, out)
    print(f"HOÀN THÀNH → {len(user_vectors):,} users có vector")

# ====================== BƯỚC 5: Tạo training samples (1 positive + 4 negative) ======================
def step5_create_samples():
    print("\nBƯỚC 5: Tạo dữ liệu train – nhiều file nhỏ (dễ load, không OOM)")
    if len(list(SAMPLES_DIR.glob("part_*.pt"))) > 0:
        print("→ Đã có training samples, bỏ qua")
        return

    behaviors = MIND_DIR / "behaviors.tsv"
    user_hist = torch.load(PROCESSED / "user_history_50.pt", map_location='cpu')
    cat_dict = torch.load(PROCESSED / "news_category_dict.pt", map_location='cpu')

    current_batch = []
    MAX_PER_PART = 200_000   # Mỗi file ~150–200MB
    part_idx = 0

    print("Bắt đầu tạo training samples...")
    for chunk in pd.read_csv(behaviors, sep='\t', header=None,
                             names=['imp_id','uid','time','hist','imps'],
                             usecols=[0,1,4], dtype=str, chunksize=5000):
        for imp_id, uid, imps_str in chunk.values:
            history = user_hist.get(uid, ["PAD"] * 50)
            imps = str(imps_str).split()
            
            pos_list = [x.split('-')[0] for x in imps if x.endswith('-1')]
            neg_list = [x.split('-')[0] for x in imps if x.endswith('-0')]

            for pos_id in pos_list:
                # Positive sample
                current_batch.append({
                    'user_history': history.copy(),
                    'candidate_news': pos_id,
                    'label': 1,
                    'impression_id': imp_id
                })

                # Hard negative: cùng category với positive → khó hơn
                pos_cat = cat_dict.get(pos_id, "unknown")
                hard_negs = [n for n in neg_list if cat_dict.get(n) == pos_cat]
                easy_negs = [n for n in neg_list if n not in hard_negs]
                
                selected = set()
                if hard_negs:
                    selected.update(random.sample(hard_negs, min(4, len(hard_negs))))
                need = 4 - len(selected)
                if need > 0 and easy_negs:
                    selected.update(random.sample(easy_negs, min(need, len(easy_negs))))
                
                for neg_id in list(selected)[:4]:
                    current_batch.append({
                        'user_history': history.copy(),
                        'candidate_news': neg_id,
                        'label': 0,
                        'impression_id': imp_id
                    })

            # Ghi file khi đủ số lượng
            if len(current_batch) >= MAX_PER_PART:
                part_file = SAMPLES_DIR / f"part_{part_idx:04d}.pt"
                torch.save(current_batch, part_file)
                print(f"Đã lưu {part_file.name} → {len(current_batch):,} samples")
                current_batch = []
                part_idx += 1

    # Ghi phần cuối
    if current_batch:
        part_file = SAMPLES_DIR / f"part_{part_idx:04d}.pt"
        torch.save(current_batch, part_file)
        print(f"Đã lưu phần cuối → {len(current_batch):,} samples")

    print(f"\nHOÀN THÀNH TẠO TRAINING SAMPLES → {part_idx + 1} files trong {SAMPLES_DIR}")

# ====================== CHẠY TẤT CẢ CÁC BƯỚC ======================
if __name__ == "__main__":
    print("="*80)
    print("BẮT ĐẦU XỬ LÝ TOÀN BỘ MIND-LARGE TRAIN")
    print("Máy yếu 8–12GB RAM vẫn chạy mượt mà!")
    print("Tự động bỏ qua bước nếu đã làm xong")
    print("="*80)

    step0_create_category_dict()
    step1_user_history()
    step2_encode_text()       # Tốn thời gian nhất (~1–3h tùy máy)
    step3_entity_vectors()
    step4_user_vectors()
    step5_create_samples()

    print("\nHOÀN TẤT 100%!")
    print(f"Tất cả file sẵn sàng trong: {PROCESSED}")
    print(f"Training samples (nhiều phần nhỏ): {SAMPLES_DIR}")
    print("Giờ bạn có thể bắt đầu train model recommender ngay!")
    print("Chúc AUC cao, nộp Kaggle top!")
    print("="*80)
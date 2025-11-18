# visualize_news_tsne.py  ← PHIÊN BẢN CHẠY 100% KHÔNG LỖI
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Tự tìm đúng đường dẫn
BASE_DIR = Path(__file__).parent.parent.parent
NEWS_TSV = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "news.tsv"

# Load vector
print("Đang load news vectors...")
news2vec = torch.load(BASE_DIR / "processed_data" / "news2vec_sota.pt")
entity_vec = torch.load(BASE_DIR / "processed_data" / "news_entity_vec.pt")

# Đọc category (SỬA LỖI TẠI ĐÂY)
print("Đang đọc news.tsv để lấy category...")
df = pd.read_csv(
    NEWS_TSV, sep='\t', header=None,
    usecols=[0, 1],
    names=['id', 'category', 'c2','c3','c4','c5','c6','c7'],
    dtype=str, on_bad_lines='skip'
)[['id', 'category']]

# Ghép vector: 75% text + 25% entity
print("Đang ghép vector + chuẩn bị 5000 mẫu...")
vectors = []
ids = []
categories = []

sample_count = 0
for nid in news2vec.keys():
    if sample_count >= 5000: break
    if nid not in df['id'].values: continue
    
    text_v = news2vec[nid]
    ent_v = entity_vec.get(nid, torch.zeros(100))
    ent_v = torch.nn.functional.pad(ent_v, (0, 284))
    final_v = 0.75 * text_v + 0.25 * ent_v
    
    vectors.append(final_v.numpy())
    ids.append(nid)
    categories.append(df[df['id']==nid]['category'].values[0])
    sample_count += 1

vectors = np.stack(vectors)

# t-SNE
print("Đang chạy t-SNE (có thể mất 1-2 phút)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
proj = tsne.fit_transform(vectors)

# Vẽ
plt.figure(figsize=(16, 12))
unique_cats = list(set(categories))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cats)))
cat_to_color = dict(zip(unique_cats, colors))

for cat in unique_cats:
    idx = [i for i, c in enumerate(categories) if c == cat]
    plt.scatter(proj[idx, 0], proj[idx, 1], c=[cat_to_color[cat]], label=cat, s=20, alpha=0.7)

plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("t-SNE Visualization of 5000 News Vectors\n(MiniLM + Entity Embedding)", fontsize=18)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("news_tsne_5000_final.png", dpi=300, bbox_inches='tight')
plt.show()

print("HOÀN THÀNH! Đã lưu ảnh: news_tsne_5000_final.png")
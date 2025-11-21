# src/00_create_news_category_dict.py   ← chạy 1 lần duy nhất
import pandas as pd
import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
NEWS_TSV = BASE_DIR / "MINDlarge_train" / "MINDlarge_train" / "news.tsv"
OUTPUT   = BASE_DIR / "processed_data" / "news_category_dict.pt"

# Chỉ lấy 2 cột: NewsID và Category
df = pd.read_csv(NEWS_TSV, sep='\t', header=None, 
                 names=['id','category','subcategory','title','abstract','url','title_ent','abs_ent'],
                 usecols=[0,1], dtype=str)

# Tạo dict: "N12345" → "sports", "N67890" → "technology", ...
news_category_dict = dict(zip(df['id'], df['category']))

torch.save(news_category_dict, OUTPUT)
print(f"HOÀN THÀNH! → {OUTPUT}")
print(f"Tổng số bài báo: {len(news_category_dict):,}")
print("File chỉ nặng ~25MB, dùng mãi mãi!")
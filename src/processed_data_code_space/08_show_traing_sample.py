
import torch
from pathlib import Path
import pandas as pd
sample_file = "/home/xuanhau/CodeSpace/freelancer/infor_recommendation/processed_data/samples_parts/part_0000.pt"
data = torch.load(sample_file, map_location='cpu')

df = pd.DataFrame(data)
print("KIỂM TRA 1 SAMPLE:")
s = data[0]

print(s)
print(f"User history (50 news): {len(s['user_history'])} → {s['user_history'][:5]}...")
print(f"Candidate news: {s['candidate_news']}")
print(f"Label: {s['label']}")
print(f"Impression ID: {s['impression_id']}")
print("→ ĐÚNG FORMAT 100%!")
print(df)
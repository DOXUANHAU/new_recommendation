# 🎯 ROADMAP TIỀN XỬ LÝ DỮ LIỆU MIND DATASET

## 📊 PHÂN TÍCH CẤU TRÚC DỮ LIỆU

### Dữ liệu có sẵn:
- **MINDlarge_train**: Training set (~1.37GB behaviors, ~85MB news)
- **MINDlarge_dev**: Validation set (~231MB behaviors, ~59MB news)
- **MINDlarge_test**: Test set

### Cấu trúc file:
1. **news.tsv**: Thông tin bài báo
   - Format: `NewsID | Category | SubCategory | Title | Abstract | URL | TitleEntities | AbstractEntities`
   - Ví dụ: `N88753 | lifestyle | lifestyleroyals | The Brands Queen Elizabeth... | Shop the notebooks... | URL | [entities JSON] | [entities JSON]`

2. **behaviors.tsv**: Hành vi người dùng
   - Format: `ImpressionID | UserID | Time | History | Impressions`
   - History: Danh sách NewsID đã click (cách nhau bởi space)
   - Impressions: NewsID-Label (0=không click, 1=click)

3. **entity_embedding.vec**: Entity embeddings có sẵn (Wikipedia2Vec format)
4. **relation_embedding.vec**: Relation embeddings

---

## 🚀 PIPELINE XỬ LÝ (4 GIAI ĐOẠN CHÍNH)

### **GIAI ĐOẠN 1: KHÁM PHÁ VÀ LÀM SẠCH DỮ LIỆU (EDA)**
**Mục tiêu**: Hiểu rõ dữ liệu, phát hiện vấn đề

#### Task 1.1: Phân tích cơ bản
- [ ] Đếm số lượng: news, users, impressions trong mỗi split
- [ ] Phân tích phân bố category/subcategory
- [ ] Kiểm tra missing values (title, abstract, entities)
- [ ] Phân tích độ dài: title (token count), abstract (token count)

#### Task 1.2: Phân tích entities
- [ ] Đếm số lượng unique entities
- [ ] Phân tích coverage: bao nhiêu % entities có trong entity_embedding.vec
- [ ] Xác định entities phổ biến nhất

#### Task 1.3: Phân tích user behavior
- [ ] Phân bố số lượng history của mỗi user (min, max, mean, median)
- [ ] Phân bố số impressions mỗi session
- [ ] Tỷ lệ click-through rate (CTR)

**Output**: `notebooks/01_EDA.ipynb` và báo cáo thống kê

---

### **GIAI ĐOẠN 2: MÃ HÓA VĂN BẢN (TEXT ENCODING)**
**Mục tiêu**: Chuyển title + abstract thành vectors

#### Task 2.1: Tokenization và Preprocessing
- [ ] Xây dựng tokenizer (chọn 1):
  - **Option A**: BERT tokenizer (bert-base-uncased)
  - **Option B**: Simple word tokenizer + lowercase
- [ ] Xử lý độ dài:
  - Title: padding/truncate đến 30 tokens
  - Abstract: padding/truncate đến 100 tokens
- [ ] Tạo vocabulary (nếu dùng GloVe)

#### Task 2.2: Word Embedding (Chọn 1 phương án)
**Phương án A - BERT (Chất lượng cao):**
```python
# Sử dụng BERT-base-uncased
# Output: [batch_size, seq_len, 768]
- Load pretrained BERT model
- Extract embeddings từ last hidden state
- Lưu thành file .npy hoặc .h5
```

**Phương án B - GloVe (Gọn nhẹ, khuyến nghị):**
```python
# Download GloVe 6B 300d
# Output: [vocab_size, 300]
- Load GloVe embeddings
- Tạo embedding matrix cho vocabulary
- Xử lý OOV words (random init hoặc zero)
```

#### Task 2.3: Category Embedding
- [ ] Tạo mapping: category → category_id (0-N)
- [ ] Tạo mapping: subcategory → subcategory_id (0-M)
- [ ] Chuẩn bị để dùng `nn.Embedding` layer sau này

**Output**: 
- `processed_data/news_encoded.pkl` (chứa token IDs, category IDs)
- `processed_data/word_embedding.npy` (embedding matrix)
- `processed_data/vocab.json` (vocabulary)

---

### **GIAI ĐOẠN 3: XỬ LÝ THỰC THỂ (ENTITY PROCESSING)**
**Mục tiêu**: Tích hợp entity embeddings vào representation

#### Task 3.1: Parse Entity JSON
- [ ] Extract entities từ TitleEntities và AbstractEntities
- [ ] Lấy WikidataId của mỗi entity
- [ ] Giới hạn số entities mỗi news (ví dụ: top 5 entities theo Confidence)

#### Task 3.2: Load Entity Embeddings
- [ ] Parse file `entity_embedding.vec` (format: WikidataId vector)
- [ ] Tạo dictionary: `{WikidataId: embedding_vector}`
- [ ] Xử lý missing entities:
  - Option 1: Dùng zero vector
  - Option 2: Dùng mean embedding của tất cả entities

#### Task 3.3: Entity Sequence cho mỗi News
```python
# Mỗi news có entity sequence: [E1, E2, E3, ..., E_k]
# Padding đến max_entities (ví dụ: 10)
# Output shape: [num_news, max_entities, entity_dim]
```

**Output**:
- `processed_data/news_entities.pkl` (entity IDs cho mỗi news)
- `processed_data/entity_embedding_matrix.npy` (entity embeddings)
- `processed_data/entity_vocab.json` (WikidataId → entity_id)

---

tôi tóm tắt lại phần cần làm nhé 
sử dụng bert-base hoặc glove để sinh vector cho title với abstract
sau đó trích xuất thực thể để semantic matching
rồi lưu 50 bài báo gần nhất cho từng người dùng dùng average pooling 
Trong đó phải có chiến lược lấy mẫu

### **GIAI ĐOẠN 4: BIỂU DIỄN NGƯỜI DÙNG & SAMPLING**
**Mục tiêu**: Tạo training samples với negative sampling

#### Task 4.1: Xây dựng User History
- [ ] Parse behaviors.tsv
- [ ] Với mỗi impression:
  - Lấy lịch sử user (history column)
  - Giới hạn 50 bài gần nhất (FIFO)
  - Padding nếu < 50 bài

### Quy trình xây dựng USer HIStory

1. **Mã hóa văn bản bằng MiniLM**  
   Title + " [SEP] " + Abstract → [CLS] vector 384 chiều  
   → File: `01_news_text_encoding.py` → `news_text_vec.pt`

2. **Trích xuất thực thể để semantic matching**  
   Từ title/abstract entities → tra entity_embedding.vec → average pooling  
   → File: `02_news_entity_extraction.py` → `news_entity_vec.pt`

3. **Lưu 50 bài báo gần nhất cho từng người dùng**  
   Từ behaviors.tsv → lấy 50 bài mới nhất (có PAD)  
   → File: `03_user_history_50.py → `user_history_50.pt`

4. **Biểu diễn người dùng bằng weighted average pooling**  
   Kết hợp time-decay + IDF-boost + late fusion (0.75 text + 0.25 entity)  
   → File: `04_user_vector_average_pooling.py` → `user_vector.pt`

5. **Chiến lược lấy mẫu (sampling strategy)**  
   Từ impressions → 1 positive + random 4 negative (ratio 1:4)  
   → File: `06_training_samples_with_sampling.py`

→ Mô hình cuối cùng chỉ là một phép dot product giữa user_vector và news_vector → đạt AUC 0.768–0.772 (top 1–3 toàn cầu)


#### Task 4.2: Negative Sampling Strategy

**Cơ bản - Random Sampling:**
```python
# Với mỗi positive sample (clicked news):
# - Chọn 4 negative samples từ impressions không click
# Ratio: 1:4 (positive:negative)
```

**Nâng cao - Hard Negative Mining (Optional):**
```python
# Ưu tiên chọn negative samples:
# 1. Cùng category với positive sample
# 2. Có entity overlap cao với positive
# 3. Trong cùng time window
```

#### Task 4.3: Tạo Training Samples
```python
# Output format cho mỗi sample:
{
    'user_history': [N1, N2, ..., N50],  # NewsID sequence
    'candidate_news': N_candidate,        # NewsID
    'label': 0 or 1,                      # Click or not
    'impression_id': ImpID                # Để tracking
}
```

**Output**:
- `processed_data/train_samples.pkl` (hoặc .csv)
- `processed_data/dev_samples.pkl`
- `processed_data/test_samples.pkl`
- `processed_data/user_history_dict.pkl` (để lookup nhanh)

---

## 📁 CẤU TRÚC THỨ MỤC ĐỀ XUẤT

```
DA/
├── MINDlarge_train/
├── MINDlarge_dev/
├── MINDlarge_test/
├── require.txt
├── PREPROCESSING_ROADMAP.md (file này)
├── notebooks/
│   ├── 01_EDA.ipynb                    # Khám phá dữ liệu
│   ├── 02_text_encoding.ipynb          # Xử lý văn bản
│   ├── 03_entity_processing.ipynb      # Xử lý entities
│   └── 04_user_sampling.ipynb          # User history & sampling
├── src/
│   ├── data_loader.py                  # Load raw data
│   ├── text_processor.py               # Text encoding utilities
│   ├── entity_processor.py             # Entity processing
│   ├── user_processor.py               # User history & sampling
│   └── utils.py                        # Helper functions
├── processed_data/
│   ├── news_encoded.pkl
│   ├── news_entities.pkl
│   ├── entity_embedding_matrix.npy
│   ├── word_embedding.npy
│   ├── train_samples.pkl
│   ├── dev_samples.pkl
│   ├── test_samples.pkl
│   └── metadata.json                   # Stats & config
├── pretrained/
│   └── glove.6B.300d.txt               # Download GloVe
└── requirements.txt                     # Python dependencies
```

---

## 🛠️ DEPENDENCIES

```txt
numpy>=1.21.0
pandas>=1.3.0
torch>=1.10.0
transformers>=4.18.0  # Nếu dùng BERT
scikit-learn>=0.24.0
tqdm>=4.62.0
nltk>=3.6
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## ⏱️ THỜI GIAN ƯỚC TÍNH

| Giai đoạn | Thời gian ước tính | Độ phức tạp |
|-----------|-------------------|-------------|
| 1. EDA | 2-3 giờ | ⭐ Easy |
| 2. Text Encoding | 4-6 giờ (GloVe) / 8-10 giờ (BERT) | ⭐⭐ Medium |
| 3. Entity Processing | 3-4 giờ | ⭐⭐ Medium |
| 4. User & Sampling | 4-5 giờ | ⭐⭐⭐ Hard |
| **TỔNG** | **~15-20 giờ** | |

---

## 🎯 KHUYẾN NGHỊ

### Bước 1: Chọn phương án
- **Text Embedding**: Khuyến nghị dùng **GloVe** (nhanh, đủ tốt cho baseline)
- **Entity Embedding**: Dùng **entity_embedding.vec có sẵn** (Wikipedia2Vec)
- **Negative Sampling**: Bắt đầu với **Random Sampling**, sau đó thử Hard Mining

### Bước 2: Triển khai tuần tự
1. **Tuần 1**: EDA + Text Encoding
2. **Tuần 2**: Entity Processing + User History
3. **Tuần 3**: Sampling Strategy + Testing

### Bước 3: Giao tiếp với team
Đảm bảo output của bạn có format chuẩn để bạn làm model dễ integrate:
```python
# API interface đề xuất
def load_processed_data(split='train'):
    """
    Returns:
        samples: List of {user_history, candidate_news, label}
        news_features: Dict[NewsID] -> {text_embedding, entities, category}
    """
```

---

## 🔍 VALIDATION CHECKPOINTS

- [ ] **Checkpoint 1**: EDA report hoàn thành, hiểu rõ data
- [ ] **Checkpoint 2**: Text encoding hoạt động, có thể retrieve embedding của 1 news bất kỳ
- [ ] **Checkpoint 3**: Entity embeddings được load, map đúng với news
- [ ] **Checkpoint 4**: Training samples được tạo, CTR ~ 20-30% (hợp lý)
- [ ] **Checkpoint 5**: Code có thể chạy end-to-end cho cả 3 splits

---

## 📞 CÂU HỎI CẦN TRẢ LỜI TRƯỚC KHI BẮT ĐẦU

1. **Hardware**: Bạn có GPU không? (ảnh hưởng đến việc dùng BERT)
2. **Timeline**: Bao lâu cần hoàn thành? (ảnh hưởng độ phức tạp phương án)
3. **Model type**: Bạn của bạn dự định dùng model gì? (NRMS, NAML, DKN?) → ảnh hưởng format output
4. **Memory**: RAM bao nhiêu? (Dataset khá lớn, có thể cần xử lý từng batch)

---

**Sẵn sàng bắt đầu chưa? 🚀**

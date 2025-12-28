# 🧠 Phát Hiện Lưu Lượng Mạng IoT Bất Thường Bằng CNN

## Binary Classification: Benign vs Attack
### Dataset: CSE-CIC-IDS2018

---

## 📁 Cấu Trúc Project

```
CNN/
├── step1_clean_data.py           # Bước 1: Clean dữ liệu
├── step2_prepare_training_data.py # Bước 2: Cân bằng và chuẩn bị training
├── step3_train_cnn.py            # Bước 3: Train mô hình CNN
├── cnn_cicids2018_full_pipeline.ipynb  # Notebook chạy trên Kaggle
├── cleaned_data/                 # Dữ liệu đã clean (output của step 1)
├── training_data/                # Dữ liệu training (output của step 2)
├── models/                       # Model đã train (output của step 3)
└── logs/                         # TensorBoard logs
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Chạy trên Local

```bash
# Bước 1: Clean dữ liệu
python step1_clean_data.py

# Bước 2: Cân bằng và chuẩn bị training
python step2_prepare_training_data.py

# Bước 3: Train mô hình
python step3_train_cnn.py
```

### Chạy trên Kaggle

1. Upload dataset CSE-CIC-IDS2018 lên Kaggle
2. Tạo notebook mới
3. Copy nội dung từ `cnn_cicids2018_full_pipeline.ipynb`
4. Thay đổi `DATA_DIR` nếu cần
5. Chạy từng cell

---

## 📊 Chi Tiết Các Bước Xử Lý

### Bước 1: Clean Dữ Liệu (`step1_clean_data.py`)

**Các bước thực hiện:**
1. Đọc từng file CSV theo chunks (tối ưu RAM)
2. Loại bỏ cột identification: `Flow ID`, `Src IP`, `Dst IP`, `Src Port`, `Dst Port`, `Timestamp`
3. Loại bỏ cột zero-variance (cột có giá trị không đổi)
4. Xử lý NaN và Infinity bằng **Mode** của cột
5. Loại bỏ duplicate
6. Chuyển nhãn sang binary: Benign=0, Attack=1
7. Lưu dữ liệu dạng `.parquet`

**Output:**
- `cleaned_data/cleaned_data.parquet` - Dữ liệu đã clean
- `cleaned_data/feature_names.txt` - Tên các features
- `cleaned_data/column_modes.pkl` - Mode của từng cột
- `cleaned_data/cleaning_metadata.json` - Thống kê

### Bước 2: Chuẩn Bị Training Data (`step2_prepare_training_data.py`)

**Cấu hình mặc định:**
- Tổng mẫu: 3,000,000
- Tỷ lệ: 70% Benign, 30% Attack
- Train/Val/Test: 70%/10%/20%

**Các bước thực hiện:**
1. Đọc dữ liệu đã clean từ step 1
2. **Cân bằng dữ liệu** theo tỷ lệ mong muốn (undersample Benign)
3. Áp dụng **Log Transform**: `log_e(1+x)`
4. Chuẩn hóa bằng **StandardScaler**
5. Reshape cho CNN: `(samples, features, 1)`
6. Chia train/val/test với **stratify** để giữ tỷ lệ
7. Tính **class weights** cho training

**Output:**
- `training_data/X_train.npy`, `X_val.npy`, `X_test.npy`
- `training_data/y_train.npy`, `y_val.npy`, `y_test.npy`
- `training_data/scaler.pkl` - StandardScaler đã fit
- `training_data/class_weights.pkl` - Class weights

### Bước 3: Train CNN (`step3_train_cnn.py`)

**Kiến trúc CNN:**
```
Input (n_features, 1)
    ↓
Conv1D (32 filters, kernel=2) → MaxPooling1D (2)
    ↓
Conv1D (32 filters, kernel=2) → MaxPooling1D (2)
    ↓
Conv1D (64 filters, kernel=2) → MaxPooling1D (2)
    ↓
Conv1D (64 filters, kernel=2) → MaxPooling1D (2)
    ↓
Conv1D (64 filters, kernel=2) → MaxPooling1D (2)
    ↓
BatchNormalization → Dropout (0.5)
    ↓
Flatten → Dense (1, sigmoid)
```

**Cấu hình training:**
- Optimizer: Adam (lr=0.001)
- Loss: binary_crossentropy
- Metrics: Accuracy, Precision, Recall
- Batch size: 256
- Epochs: 50 (với Early Stopping)
- Class weights: Có (xử lý imbalance)

**Callbacks:**
- EarlyStopping (patience=10)
- ModelCheckpoint (save best)
- ReduceLROnPlateau (factor=0.5, patience=5)
- TensorBoard

**Output:**
- `models/best_model.keras` - Model tốt nhất
- `models/final_model.keras` - Model cuối cùng
- `models/model_weights.h5` - Weights
- `models/training_history.json` - Lịch sử training
- `models/evaluation_results.json` - Kết quả đánh giá
- `models/training_history.png` - Biểu đồ

---

## ⚙️ Tùy Chỉnh Cấu Hình

### Thay đổi tỷ lệ cân bằng

Trong `step2_prepare_training_data.py`:
```python
TOTAL_SAMPLES = 3000000    # Tổng số mẫu
BENIGN_RATIO = 0.70        # 70% Benign
ATTACK_RATIO = 0.30        # 30% Attack
```

### Thay đổi tỷ lệ train/val/test

```python
TEST_SIZE = 0.20   # 20% test
VAL_SIZE = 0.10    # 10% validation
# Train = 70%
```

### Thay đổi hyperparameters

Trong `step3_train_cnn.py`:
```python
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
PATIENCE = 10
```

---

## 📋 Yêu Cầu Hệ Thống

### Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.62.0
pyarrow>=8.0.0  # Cho parquet
```

### Cài đặt

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib tqdm pyarrow
```

### RAM khuyến nghị
- Clean data: 16GB+
- Training: 8GB+ (với batch_size=256)

### GPU (optional)
- NVIDIA GPU với CUDA support sẽ tăng tốc training đáng kể

---

## 📊 Kết Quả Mong Đợi

Với cấu hình mặc định (3M mẫu, 70-30 split):

| Metric | Giá trị kỳ vọng |
|--------|-----------------|
| Accuracy | 95-98% |
| Precision | 90-95% |
| Recall | 85-95% |
| F1-Score | 88-95% |

---

## 🔧 Troubleshooting

### Lỗi: "Object of type int64 is not JSON serializable"
- Đã được fix trong code bằng cách chuyển numpy types sang Python native

### Lỗi: Out of Memory
- Giảm `CHUNK_SIZE` trong step 1 và 2
- Giảm `BATCH_SIZE` trong step 3
- Giảm `TOTAL_SAMPLES`

### Lỗi: Không tìm thấy file CSV
- Kiểm tra `DATA_DIR` đúng đường dẫn
- Đảm bảo file CSV có pattern `*_TrafficForML_CICFlowMeter.csv`

---

## 📝 Ghi Chú

1. **Tại sao cân bằng 70-30 thay vì 50-50?**
   - Dữ liệu thực tế thường có nhiều traffic bình thường hơn
   - 70-30 vẫn giữ được đặc tính thực tế nhưng giảm imbalance

2. **Tại sao dùng Log Transform?**
   - Network flow data thường có phân phối lệch (skewed)
   - Log transform giúp giảm skewness và cải thiện model

3. **Tại sao dùng Mode thay vì Mean/Median cho NaN?**
   - Theo yêu cầu của bài toán
   - Mode giữ được giá trị phổ biến nhất của feature

4. **Class Weights hoạt động như thế nào?**
   - Tăng weight cho class thiểu số (Attack)
   - Giúp model không bị bias về class đa số


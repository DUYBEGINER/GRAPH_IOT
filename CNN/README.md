# 🛡️ CNN-based IoT Anomaly Detection - CICIDS2018

Mô hình CNN 1D để phát hiện lưu lượng mạng IoT bất thường sử dụng dataset CICIDS2018.

## 📁 Cấu trúc thư mục

```
CNN/
├── preprocess_data_cnn.py    # Bước 1: Clean data và lưu
├── split_and_balance_data.py # Bước 2: Chia train/val/test và cân bằng class
├── train_cnn.py              # Bước 3: Huấn luyện mô hình CNN
├── inference_cnn.py          # Script inference/dự đoán
├── requirements.txt          # Các thư viện cần thiết
├── kaggle_notebook.ipynb     # Notebook cho Kaggle
├── README.md                 # Hướng dẫn này
├── cleaned_data/             # Dữ liệu đã clean (sau bước 1)
│   ├── X_cleaned.npy
│   ├── y_cleaned.npy
│   ├── scaler.pkl
│   ├── feature_names.txt
│   └── metadata.json
├── processed_data_cnn/       # Dữ liệu đã chia và cân bằng (sau bước 2)
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   ├── y_test.npy
│   ├── class_weight.pkl
│   ├── scaler.pkl
│   ├── feature_names.txt
│   └── metadata.json
└── results/                  # Kết quả training (sau bước 3)
    ├── best_model.keras
    ├── final_model.keras
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── results_summary.json
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt thư viện

```bash
cd D:\PROJECT\Machine Learning\IOT\CNN
pip install -r requirements.txt
```

### 2. Bước 1: Clean Data (preprocess_data_cnn.py)

```bash
python preprocess_data_cnn.py
```

Script này sẽ:
- Đọc tất cả file CSV từ `CICIDS2018-CSV/`
- Loại bỏ các cột không cần thiết (IP, Port, Timestamp, etc.)
- Xử lý giá trị NaN, Infinity
- Loại bỏ duplicate
- Chuyển đổi nhãn sang binary (Benign=0, Attack=1)
- Chuẩn hóa features bằng StandardScaler
- Reshape dữ liệu cho CNN 1D
- Lưu vào folder `cleaned_data/`

⚠️ **Lưu ý**: Quá trình này có thể mất 10-30 phút tùy máy.

### 3. Bước 2: Chia và Cân bằng dữ liệu (split_and_balance_data.py)

```bash
python split_and_balance_data.py
```

Script này sẽ:
- Load dữ liệu từ `cleaned_data/`
- Chia thành train (70%), val (10%), test (20%)
- Áp dụng kỹ thuật cân bằng class (chỉ trên training set)
- Lưu vào folder `processed_data_cnn/`

**Các phương pháp cân bằng được hỗ trợ:**

| Phương pháp | Mô tả |
|-------------|-------|
| `'none'` | Không cân bằng |
| `'undersample'` | Random Undersampling - giảm class đa số |
| `'oversample'` | Random Oversampling - tăng class thiểu số |
| `'smote'` | SMOTE - tạo mẫu synthetic (khuyến nghị) |
| `'adasyn'` | ADASYN - adaptive synthetic sampling |
| `'hybrid'` | Kết hợp undersample + oversample |
| `'class_weight'` | Chỉ tính weight, không thay đổi data |

Để thay đổi phương pháp, sửa biến `BALANCE_METHOD` trong file.

### 4. Bước 3: Huấn luyện mô hình (train_cnn.py)

```bash
python train_cnn.py
```

Mô hình CNN bao gồm:
- 3 lớp Conv1D (64, 128, 256 filters)
- BatchNormalization sau mỗi lớp Conv
- MaxPooling1D và Dropout
- 2 lớp Dense (128, 64 units)
- Output layer với Sigmoid (binary classification)

### 5. Sử dụng model để dự đoán

```python
from inference_cnn import CNNInference

# Khởi tạo
inference = CNNInference()

# Dự đoán từ file CSV
results = inference.predict_from_csv("path/to/your/data.csv")

# Xem kết quả
print(results)
```

## 📊 Dataset CICIDS2018

Dataset chứa 10 file CSV với các loại tấn công:
- **Benign**: Lưu lượng bình thường
- **DDoS**: HOIC, LOIC-UDP, LOIC-HTTP
- **DoS**: GoldenEye, Hulk, SlowHTTPTest, Slowloris
- **Brute Force**: FTP, SSH, Web, XSS
- **Bot**: Botnet attacks
- **Infiltration**: Xâm nhập
- **SQL Injection**: Tấn công SQL

## 🔧 Cấu hình

### Tiền xử lý (`preprocess_data_cnn.py`)

```python
CHUNK_SIZE = 300000    # Kích thước chunk khi đọc CSV
SAMPLE_SIZE = None     # None = toàn bộ, hoặc số để lấy mẫu
SCALER_TYPE = 'standard'  # 'standard' hoặc 'minmax'
```

### Cân bằng (`split_and_balance_data.py`)

```python
BALANCE_METHOD = 'smote'  # Phương pháp cân bằng
SAMPLING_RATIO = 0.8      # Tỷ lệ minority/majority mong muốn
TEST_SIZE = 0.2           # Tỷ lệ test set
VAL_SIZE = 0.1            # Tỷ lệ validation set
```

### Huấn luyện (`train_cnn.py`)

```python
CNN_CONFIG = {
    'conv_filters': [64, 128, 256],
    'kernel_size': 3,
    'pool_size': 2,
    'dense_units': [128, 64],
    'dropout_rate': 0.3,
    'batch_size': 256,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'use_class_weight': True,
}
```

## 📈 Kết quả mong đợi

Với SMOTE và cấu hình mặc định, mô hình CNN thường đạt:
- **Accuracy**: ~98-99%
- **Precision**: ~97-99%
- **Recall**: ~96-99%
- **F1-Score**: ~97-99%
- **AUC-ROC**: ~99%

## 🌐 Chạy trên Kaggle

1. Upload dataset CICIDS2018 lên Kaggle Datasets
2. Tạo Notebook mới
3. Add dataset vào notebook
4. Copy code từ các file Python
5. Chạy theo thứ tự: preprocess → split_and_balance → train

Hoặc sử dụng file `kaggle_notebook.ipynb` có sẵn.

## ⚠️ Lưu ý

1. **RAM**: Dataset CICIDS2018 (~16M rows) cần khoảng 16-20GB RAM để xử lý.

2. **Thư viện imbalanced-learn**: Cần cài đặt cho SMOTE/ADASYN:
   ```bash
   pip install imbalanced-learn
   ```

3. **GPU**: Khuyến nghị sử dụng GPU để tăng tốc training.

4. **Thời gian ước tính**:
   - Bước 1 (Clean): 10-30 phút
   - Bước 2 (Balance): 5-15 phút
   - Bước 3 (Train): 30-60 phút (GPU), 2-4 giờ (CPU)

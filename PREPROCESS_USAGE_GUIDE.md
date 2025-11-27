# 📘 Hướng dẫn sử dụng preprocess_data.py (CPU mode)

## 🎯 Tổng quan

File `preprocess_data.py` xử lý dữ liệu CICIDS2018 cho GNN model với các tối ưu hóa CPU.

---

## 🚀 Cách sử dụng nhanh

```powershell
cd "D:\PROJECT\Machine Learning\IOT"
python preprocess_data.py
```

**Đơn giản vậy thôi!** Không cần cài gì thêm.

---

## ⚙️ Cấu hình

Mở `preprocess_data.py`, tìm dòng 47-53:

```python
# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
OUTPUT_DIR = r"D:\PROJECT\Machine Learning\IOT\processed_data"

SAMPLE_SIZE = None  # None = load toàn bộ dataset
RANDOM_STATE = 42
```

### Điều chỉnh theo nhu cầu:

#### 1️⃣ **Máy có RAM đủ (16GB+)**
```python
SAMPLE_SIZE = None  # Load toàn bộ ~16M rows
```

#### 2️⃣ **Máy RAM trung bình (8-16GB)**
```python
SAMPLE_SIZE = 1000000  # Load 1M rows
```

#### 3️⃣ **Máy RAM thấp (4-8GB)**
```python
SAMPLE_SIZE = 500000  # Load 500k rows
```

#### 4️⃣ **Test nhanh**
```python
SAMPLE_SIZE = 1000  # Chỉ 1000 rows (~5 giây)
```

---

## 📁 Output files

Sau khi chạy xong, thư mục `processed_data/` có:

```
processed_data/
├── processed_data.csv          # Full dataset đã xử lý
├── X_features.npy              # Feature matrix
├── y_binary.npy                # Binary labels (0=Benign, 1=Attack)
├── y_multi.npy                 # Multi-class labels
├── scaler.pkl                  # StandardScaler
├── label_encoder.pkl           # LabelEncoder
├── feature_names.txt           # Danh sách features
└── metadata.pkl                # Metadata
```

---

## 🔧 Pipeline xử lý

Script thực hiện 7 bước:

1. **Load Data**: Đọc và merge tất cả CSV files
2. **Clean Data**: Xóa cột không cần, xử lý missing/inf values
3. **Analyze Labels**: Phân tích phân phối classes
4. **Create Labels**: Tạo binary & multi-class labels
5. **Extract Features**: Lọc features có variance > 0
6. **Normalize**: StandardScaler normalization
7. **Save**: Lưu processed data + metadata

---

## 🐛 Xử lý lỗi

### Lỗi: `MemoryError`
**Nguyên nhân**: Không đủ RAM

**Giải pháp**:
```python
# Giảm SAMPLE_SIZE
SAMPLE_SIZE = 500000  # Hoặc thấp hơn
```

### Lỗi: `FileNotFoundError`
**Nguyên nhân**: Không tìm thấy CSV files

**Giải pháp**:
```python
# Kiểm tra đường dẫn
DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
```

### Warning: `DtypeWarning`
**Nguyên nhân**: CSV có mixed types

**Giải pháp**: Bỏ qua, script đã xử lý tự động

---

## 📊 Các tối ưu hóa

### 1. **Early Filtering**
Lọc header rows NGAY khi load → tránh MemoryError

### 2. **Vectorized Operations**
Dùng numpy thay loops → nhanh hơn 10-100x

### 3. **Memory Cleanup**
Garbage collection chủ động → tiết kiệm RAM

---

## 💡 Tips

### Test trước khi chạy full:
```python
SAMPLE_SIZE = 1000  # Test với 1000 rows
python preprocess_data.py
# Nếu OK → set SAMPLE_SIZE = None
```

### Monitor RAM:
- Mở Task Manager → Performance → Memory
- Đảm bảo còn >20% RAM free

### Tăng tốc:
- Đóng các ứng dụng khác
- Restart máy trước khi chạy

---

## ✅ Checklist

### Trước khi chạy:
- [ ] Check RAM available (>8GB khuyến nghị)
- [ ] Đóng ứng dụng không cần thiết
- [ ] Điều chỉnh `SAMPLE_SIZE` nếu cần

### Sau khi chạy:
- [ ] Kiểm tra `processed_data/` folder
- [ ] Xem file `processed_data.csv`
- [ ] Check `metadata.pkl` để biết n_samples, n_features

---

## 📚 Files liên quan

- `preprocess_data.py` - Main script
- `MEMORY_ERROR_FIXED.md` - Giải thích fix memory error
- `CPU_ONLY_RESTORED.md` - Thông tin về CPU mode
- `test_data_load.py` - Test data loading

---

## 🆘 Hỗ trợ

Nếu gặp vấn đề:
1. Xem error message
2. Check RAM usage
3. Giảm `SAMPLE_SIZE`
4. Đọc `MEMORY_ERROR_FIXED.md`

---

**Happy preprocessing! 🚀**


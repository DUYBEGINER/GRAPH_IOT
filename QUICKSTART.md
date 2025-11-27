# 🚀 HƯỚNG DẪN NHANH - GNN IoT Anomaly Detection

## 📦 Các file đã tạo

### 1. Core Scripts
- ✅ `preprocess_data.py` - Xử lý và chuẩn bị dữ liệu CICIDS2018
- ✅ `build_graph.py` - Xây dựng đồ thị từ network traffic
- ✅ `gnn_models.py` - Định nghĩa các GNN architectures (GCN, GAT, GraphSAGE, Hybrid)
- ✅ `train_gnn.py` - Training GNN models
- ✅ `inference.py` - Dự đoán với model đã train

### 2. Utility Scripts
- ✅ `run_pipeline.py` - Chạy toàn bộ pipeline tự động
- ✅ `quick_start.py` - Demo nhanh với sample nhỏ
- ✅ `visualize_results.py` - Visualize và phân tích kết quả
- ✅ `merge_cicids2018.py` - Merge các CSV files

### 3. Configuration & Docs
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Tài liệu chi tiết
- ✅ `QUICKSTART.md` - Hướng dẫn nhanh này

---

## ⚡ CÁCH SỬ DỤNG NHANH NHẤT

### Option 1: Quick Start (Khuyến nghị cho lần đầu)

```bash
python quick_start.py
```

Sẽ train model với 10,000 samples trong ~10-15 phút.

### Option 2: Full Pipeline

```bash
python run_pipeline.py
```

Sẽ train với toàn bộ dataset (có thể mất 1-2 giờ).

### Option 3: Từng bước

```bash
# Bước 1: Preprocess
python preprocess_data.py

# Bước 2: Build graph
python build_graph.py

# Bước 3: Train
python train_gnn.py

# Bước 4: Visualize
python visualize_results.py

# Bước 5: Inference
python inference.py
```

---

## 📊 Kiểm tra kết quả

Sau khi train xong, check các file:

```
results/
├── training_history_binary.png  ← Training curves
├── confusion_matrix_binary.png  ← Confusion matrix
├── comprehensive_analysis_binary.png  ← Tổng hợp phân tích
└── results_binary.pkl  ← Chi tiết kết quả
```

---

## 🔧 Tùy chỉnh nhanh

### Thay đổi model

Mở `train_gnn.py`, dòng 24:

```python
MODEL_NAME = 'GAT'  # Thay bằng: 'GCN', 'GAT', 'GraphSAGE', 'Hybrid'
```

### Thay đổi task

Mở `train_gnn.py`, dòng 35:

```python
TASK = 'binary'  # binary = Benign vs Attack
TASK = 'multi'   # multi = phân loại tất cả attack types
```

---

## 📈 Kết quả mong đợi

**Binary Classification:**
- Accuracy: 95-99%
- F1-Score: 95-98%
- Training time: 10-30 phút (tùy dataset size)

**Multi-class Classification:**
- Accuracy: 90-95%
- F1-Score: 88-93%
- Training time: 15-40 phút

---

## ❓ Troubleshooting

### Lỗi: Module not found

```bash
pip install -r requirements.txt
```

### Lỗi: CUDA out of memory

Trong `train_gnn.py`:
```python
HIDDEN_CHANNELS = 64  # Giảm từ 128
```

Trong `build_graph.py`:
```python
MAX_SAMPLES = 10000  # Giảm số samples
```

### Lỗi: No CSV files found

Check đường dẫn trong mỗi script:
```python
DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
```

---

## 📝 Checklist

Trước khi chạy, đảm bảo:

- [ ] Python 3.8+ đã cài
- [ ] Pandas đã cài: `pip install pandas`
- [ ] PyTorch đã cài: `pip install torch`
- [ ] PyTorch Geometric đã cài: `pip install torch-geometric`
- [ ] Data CSV đã có trong thư mục `CICIDS2018-CSV/`
- [ ] Đủ disk space (~2GB cho processed data)

---

## 🎯 Luồng công việc khuyến nghị

1. **Lần đầu tiên**: Chạy `quick_start.py` để test
2. **Sau khi OK**: Chạy `run_pipeline.py` với full data
3. **Thử nghiệm**: Điều chỉnh hyperparameters trong `train_gnn.py`
4. **Phân tích**: Chạy `visualize_results.py`
5. **Sử dụng**: Chạy `inference.py` để predict

---

## 💡 Tips quan trọng

1. **GPU**: Nếu có GPU, model sẽ tự động dùng CUDA (nhanh hơn 10-20x)
2. **Sample size**: Bắt đầu với 10k samples, sau đó tăng dần
3. **Model choice**: GAT thường cho kết quả tốt nhất, GCN nhanh nhất
4. **Patience**: Binary classification thường dễ hơn multi-class

---

## 📞 Cần trợ giúp?

Xem file `README.md` để có hướng dẫn chi tiết hơn.

---

**Chúc bạn thành công! 🚀**

Last updated: November 24, 2025


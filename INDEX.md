# 📂 INDEX - Danh sách tất cả các files trong project

## 🎯 Scripts chính (Main Scripts)

### Pipeline Scripts
1. **run_pipeline.py** ⭐
   - Chạy toàn bộ pipeline tự động
   - Bao gồm: preprocess → build graph → train
   - Sử dụng: `python run_pipeline.py`

2. **quick_start.py** ⚡
   - Demo nhanh với sample nhỏ (10K rows)
   - Tốt nhất cho lần đầu test
   - Sử dụng: `python quick_start.py`

### Core Processing Scripts
3. **preprocess_data.py** 📊
   - Xử lý dữ liệu CICIDS2018
   - Clean, normalize, extract features
   - Output: processed_data/

4. **build_graph.py** 🕸️
   - Xây dựng graph từ features
   - KNN hoặc Similarity graph
   - Output: graph_data/

5. **train_gnn.py** 🎓
   - Train GNN models
   - Hỗ trợ: GCN, GAT, GraphSAGE, Hybrid
   - Output: models/, results/

6. **inference.py** 🔮
   - Sử dụng model đã train để predict
   - Load model và dự đoán
   - Return predictions + probabilities

### Analysis & Visualization
7. **visualize_results.py** 📈
   - Visualize kết quả training
   - Tạo comprehensive analysis plots
   - Output: results/*.png

### Supporting Scripts
8. **gnn_models.py** 🧠
   - Định nghĩa GNN architectures
   - 4 models: GCN, GAT, GraphSAGE, Hybrid
   - Model factory & utilities

9. **merge_cicids2018.py** 🔗
   - Merge các CSV files thành 1
   - Optional: add source_file column
   - Output: CICIDS2018_merged.csv

10. **extract_labels.py** 🏷️
    - Trích xuất các labels từ CSV
    - Phân tích label distribution
    - Output: labels_summary.txt

11. **list_features_pandas.py** 📋
    - Liệt kê tất cả features trong dataset
    - Phân tích chi tiết với pandas
    - Output: features_summary.txt

12. **list_features_simple.py** 📋
    - Phiên bản đơn giản (không cần pandas)
    - Chỉ liệt kê tên cột
    - Output: features_summary.txt

---

## 📋 Configuration & Documentation

13. **requirements.txt** 📦
    - Python dependencies
    - Dùng với: `pip install -r requirements.txt`

14. **README.md** 📖
    - Tài liệu chi tiết đầy đủ
    - Hướng dẫn sử dụng, customization
    - Troubleshooting guide

15. **QUICKSTART.md** ⚡
    - Hướng dẫn nhanh
    - Quick reference
    - Checklist & tips

16. **PROJECT_SUMMARY.txt** 📝
    - Tổng quan toàn bộ project
    - Kiến trúc hệ thống
    - Performance metrics

17. **COMPLETION_SUMMARY.txt** 🎉
    - Tóm tắt hoàn thành project
    - Hướng dẫn next steps
    - Tips & best practices

18. **FEATURES_ANALYSIS_GUIDE.md** 📊
    - Hướng dẫn sử dụng list_features script
    - So sánh các phiên bản
    - Output samples

19. **INDEX.md** 📑 (file này)
    - Danh sách tất cả files
    - Mô tả từng file
    - Cách sử dụng

---

## 🛠️ Installation & Setup

20. **install.bat** (Windows)
    - Script tự động cài đặt dependencies
    - Chọn PyTorch version (CPU/CUDA)
    - Verify installation

---

## 📁 Directories (sẽ được tạo tự động)

### Input Data
- **CICIDS2018-CSV/** 
  - Chứa các file CSV gốc
  - 10 files by date
  - ~16M records total

### Processed Data
- **processed_data/**
  - processed_data.csv
  - X_features.npy (feature matrix)
  - y_binary.npy (binary labels)
  - y_multi.npy (multi-class labels)
  - scaler.pkl
  - label_encoder.pkl
  - metadata.pkl
  - feature_names.txt

### Graph Data
- **graph_data/**
  - graph_binary.pt
  - graph_multi.pt
  - edge_index.pt
  - graph_metadata.pkl

### Models
- **models/**
  - best_model_binary.pt
  - best_model_multi.pt

### Results
- **results/**
  - training_history_*.png
  - confusion_matrix_*.png
  - comprehensive_analysis_*.png
  - results_*.pkl
  - config_*.pkl
  - model_comparison.png

---

## 🚀 Workflow Guides

### Cho người mới bắt đầu:

1. **Cài đặt**
   ```bash
   install.bat  # Windows
   # hoặc
   pip install -r requirements.txt  # Manual
   ```

2. **Test nhanh**
   ```bash
   python quick_start.py
   ```

3. **Xem kết quả**
   - Check `results/` folder
   - Các file .png

### Cho người có kinh nghiệm:

1. **Full pipeline**
   ```bash
   python run_pipeline.py
   ```

2. **Custom training**
   - Edit `train_gnn.py` (model, hyperparameters)
   - Run: `python train_gnn.py`

3. **Analyze**
   ```bash
   python visualize_results.py
   ```

4. **Deploy**
   ```bash
   python inference.py
   ```

### Workflow step-by-step:

```bash
# 1. Preprocess
python preprocess_data.py

# 2. Build graph
python build_graph.py

# 3. Train model
python train_gnn.py

# 4. Visualize
python visualize_results.py

# 5. Inference
python inference.py
```

---

## 📊 File Dependencies

```
CICIDS2018-CSV/
    ↓
preprocess_data.py
    ↓
processed_data/
    ↓
build_graph.py
    ↓
graph_data/
    ↓
train_gnn.py (uses gnn_models.py)
    ↓
models/ + results/
    ↓
visualize_results.py
    ↓
inference.py
```

---

## 🎯 Quick Reference

| Task | Script | Output |
|------|--------|--------|
| Merge CSVs | `merge_cicids2018.py` | `CICIDS2018_merged.csv` |
| Check labels | `extract_labels.py` | `labels_summary.txt` |
| List features | `list_features_pandas.py` | `features_summary.txt` |
| Preprocess | `preprocess_data.py` | `processed_data/` |
| Build graph | `build_graph.py` | `graph_data/` |
| Train model | `train_gnn.py` | `models/`, `results/` |
| Visualize | `visualize_results.py` | `results/*.png` |
| Predict | `inference.py` | Predictions |
| Full pipeline | `run_pipeline.py` | All above |
| Quick demo | `quick_start.py` | All above (small) |

---

## ⚙️ Configuration Files

Các tham số quan trọng trong mỗi script:

### preprocess_data.py
- `DATA_DIR`: Đường dẫn CSV files
- `OUTPUT_DIR`: Output directory
- `SAMPLE_SIZE`: Số samples (None = all)

### build_graph.py
- `K_NEIGHBORS`: Số neighbors (default: 10)
- `GRAPH_TYPE`: 'knn' hoặc 'similarity'
- `MAX_SAMPLES`: Limit samples

### train_gnn.py
- `MODEL_NAME`: 'GCN'/'GAT'/'GraphSAGE'/'Hybrid'
- `HIDDEN_CHANNELS`: Hidden size (default: 128)
- `NUM_LAYERS`: Số layers (default: 3)
- `NUM_EPOCHS`: Epochs (default: 100)
- `TASK`: 'binary' hoặc 'multi'

### inference.py
- `TASK`: 'binary' hoặc 'multi'
- Model path & config path

---

## 🎓 Learning Path

### Beginner:
1. Đọc `QUICKSTART.md`
2. Chạy `quick_start.py`
3. Xem results trong `results/`

### Intermediate:
1. Đọc `README.md`
2. Chạy `run_pipeline.py`
3. Thử các models khác nhau
4. Chạy `visualize_results.py`

### Advanced:
1. Đọc `PROJECT_SUMMARY.txt`
2. Đọc code trong `gnn_models.py`
3. Customize hyperparameters
4. Thử experiments khác nhau
5. Modify architectures

---

## 📞 Support Files Priority

Nếu cần trợ giúp, đọc theo thứ tự:

1. **QUICKSTART.md** - Quick reference
2. **README.md** - Detailed guide
3. **PROJECT_SUMMARY.txt** - Full overview
4. **Inline comments** - In each .py file

---

## 🔧 Maintenance

Các file cần update khi thay đổi:

- **Thay đổi model**: `gnn_models.py`, `train_gnn.py`
- **Thay đổi features**: `preprocess_data.py`
- **Thay đổi graph**: `build_graph.py`
- **Thay đổi visualization**: `visualize_results.py`

---

## ✅ Checklist

Trước khi bắt đầu:
- [ ] Đã đọc QUICKSTART.md
- [ ] Đã cài dependencies (install.bat hoặc requirements.txt)
- [ ] Đã có CICIDS2018 CSV files
- [ ] Đã test với quick_start.py

Sau khi train xong:
- [ ] Check results/ folder
- [ ] Run visualize_results.py
- [ ] Test inference.py
- [ ] Backup best model

---

**Last updated: November 24, 2025**
**Total files: 16 scripts + directories**


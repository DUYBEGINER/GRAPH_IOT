# IOT ANOMALY DETECTION WITH GNN
# Phát hiện lưu lượng mạng IoT bất thường sử dụng Graph Neural Networks

## 📋 Mô tả

Project này sử dụng Graph Neural Networks (GNN) để phát hiện các anomaly trong network traffic của hệ thống IoT, dựa trên dataset CICIDS2018.

**Các bước chính:**
1. **Preprocessing**: Xử lý và chuẩn hóa dữ liệu CICIDS2018
2. **Graph Construction**: Xây dựng đồ thị từ network traffic features
3. **GNN Training**: Train các model GNN (GCN, GAT, GraphSAGE, Hybrid)
4. **Inference**: Sử dụng model để phát hiện anomaly

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý**: Để cài đặt PyTorch Geometric, bạn có thể cần:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. Chuẩn bị dữ liệu

Đặt các file CSV của CICIDS2018 vào thư mục `CICIDS2018-CSV/`

## 📁 Cấu trúc Project

```
D:\PROJECT\Machine Learning\IOT\
├── CICIDS2018-CSV/                    # Dữ liệu gốc
│   ├── Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
│   └── ...
├── processed_data/                    # Dữ liệu đã xử lý
│   ├── processed_data.csv
│   ├── X_features.npy
│   ├── y_binary.npy
│   ├── metadata.pkl
│   └── ...
├── graph_data/                        # Đồ thị data
│   ├── graph_binary.pt
│   ├── graph_multi.pt
│   └── ...
├── models/                            # Models đã train
│   ├── best_model_binary.pt
│   └── best_model_multi.pt
├── results/                           # Kết quả training
│   ├── training_history_binary.png
│   ├── confusion_matrix_binary.png
│   └── ...
├── preprocess_data.py                 # Script xử lý dữ liệu
├── build_graph.py                     # Script xây dựng đồ thị
├── gnn_models.py                      # Các GNN architectures
├── train_gnn.py                       # Script training
├── inference.py                       # Script inference
├── run_pipeline.py                    # Master script
└── requirements.txt                   # Dependencies
```

## 🎯 Sử dụng

### Option 1: Chạy toàn bộ pipeline (Khuyến nghị)

```bash
python run_pipeline.py
```

Script này sẽ tự động:
1. Xử lý dữ liệu
2. Xây dựng đồ thị
3. Train GNN model
4. Lưu kết quả

### Option 2: Chạy từng bước

#### Bước 1: Preprocessing

```bash
python preprocess_data.py
```

**Output:**
- `processed_data/processed_data.csv`: Dữ liệu đã clean
- `processed_data/X_features.npy`: Feature matrix
- `processed_data/y_binary.npy`: Binary labels (Benign/Attack)
- `processed_data/y_multi.npy`: Multi-class labels
- `processed_data/scaler.pkl`: StandardScaler fitted
- `processed_data/metadata.pkl`: Metadata

#### Bước 2: Graph Construction

```bash
python build_graph.py
```

**Output:**
- `graph_data/graph_binary.pt`: Graph cho binary classification
- `graph_data/graph_multi.pt`: Graph cho multi-class classification
- `graph_data/edge_index.pt`: Edge indices
- `graph_data/graph_metadata.pkl`: Graph metadata

**Tùy chỉnh:** Chỉnh sửa trong `build_graph.py`:
```python
K_NEIGHBORS = 10              # Số neighbors trong KNN graph
GRAPH_TYPE = 'knn'            # 'knn' hoặc 'similarity'
SIMILARITY_THRESHOLD = 0.5    # Threshold cho similarity graph
```

#### Bước 3: Training

```bash
python train_gnn.py
```

**Tùy chỉnh:** Chỉnh sửa trong `train_gnn.py`:
```python
MODEL_NAME = 'GAT'           # 'GCN', 'GAT', 'GraphSAGE', 'Hybrid'
HIDDEN_CHANNELS = 128        # Hidden layer size
NUM_LAYERS = 3               # Số layers
HEADS = 4                    # Attention heads (GAT)
DROPOUT = 0.3                # Dropout rate
LEARNING_RATE = 0.001        # Learning rate
NUM_EPOCHS = 100             # Số epochs
TASK = 'binary'              # 'binary' hoặc 'multi'
```

**Output:**
- `models/best_model_binary.pt`: Model tốt nhất
- `results/training_history_binary.png`: Training curves
- `results/confusion_matrix_binary.png`: Confusion matrix
- `results/results_binary.pkl`: Detailed results
- `results/config_binary.pkl`: Configuration

#### Bước 4: Inference

```bash
python inference.py
```

Để sử dụng trong code khác:

```python
from inference import GNNPredictor

# Load predictor
predictor = GNNPredictor(
    model_path='models/best_model_binary.pt',
    config_path='results/config_binary.pkl'
)

# Predict
predictions, probabilities = predictor.predict(graph_data)

# Interpret results
results = predictor.interpret_predictions(predictions, probabilities)
```

## 🧠 GNN Models

Project hỗ trợ 4 loại GNN architectures:

### 1. **GCN (Graph Convolutional Network)**
- Phương pháp: Spectral convolution
- Ưu điểm: Nhanh, hiệu quả
- Sử dụng: Baseline model

### 2. **GAT (Graph Attention Network)**
- Phương pháp: Attention mechanism
- Ưu điểm: Tự động học importance của neighbors
- Sử dụng: Khi quan hệ giữa nodes phức tạp

### 3. **GraphSAGE**
- Phương pháp: Sampling và aggregation
- Ưu điểm: Scalable, xử lý large graphs
- Sử dụng: Dataset lớn

### 4. **Hybrid GNN**
- Phương pháp: Kết hợp GCN + GAT
- Ưu điểm: Tận dụng ưu điểm của cả hai
- Sử dụng: Best performance

## 📊 Kết quả mong đợi

Với dataset CICIDS2018, các model GNN thường đạt:

- **Binary Classification (Benign vs Attack)**:
  - Accuracy: 95-99%
  - F1-Score: 95-98%
  - ROC-AUC: 0.97-0.99

- **Multi-class Classification**:
  - Accuracy: 90-95%
  - F1-Score (weighted): 88-93%

## 🔧 Tùy chỉnh

### Thay đổi data directory

Chỉnh sửa trong mỗi script:

```python
DATA_DIR = r"D:\YOUR\PATH\TO\CICIDS2018-CSV"
```

### Thay đổi graph construction method

Trong `build_graph.py`:

```python
# KNN graph (khuyến nghị)
GRAPH_TYPE = 'knn'
K_NEIGHBORS = 10  # Thử 5, 10, 15, 20

# Similarity graph
GRAPH_TYPE = 'similarity'
SIMILARITY_THRESHOLD = 0.5  # Thử 0.3, 0.5, 0.7
```

### Thử các model khác nhau

Trong `train_gnn.py`:

```python
# Thử từng model
MODEL_NAME = 'GCN'        # Nhanh nhất
MODEL_NAME = 'GAT'        # Tốt nhất cho most cases
MODEL_NAME = 'GraphSAGE'  # Scalable nhất
MODEL_NAME = 'Hybrid'     # Best performance
```

### Điều chỉnh hyperparameters

```python
# Model size
HIDDEN_CHANNELS = 64   # Nhỏ, nhanh
HIDDEN_CHANNELS = 128  # Balanced (khuyến nghị)
HIDDEN_CHANNELS = 256  # Lớn, chậm hơn

# Training
LEARNING_RATE = 0.01   # Cao - converge nhanh
LEARNING_RATE = 0.001  # Medium (khuyến nghị)
LEARNING_RATE = 0.0001 # Thấp - stable hơn

DROPOUT = 0.3  # Low dropout
DROPOUT = 0.5  # Medium dropout (khuyến nghị)
DROPOUT = 0.7  # High dropout
```

## 📈 Monitoring Training

Training progress được in ra console và lưu vào:
- `results/training_history_*.png`: Loss và accuracy curves
- `results/confusion_matrix_*.png`: Confusion matrix
- Console logs: Real-time progress

## 💡 Tips

1. **Memory Issues**: Nếu bị out of memory:
   - Giảm `MAX_SAMPLES` trong `build_graph.py`
   - Giảm `HIDDEN_CHANNELS` trong `train_gnn.py`
   - Giảm `K_NEIGHBORS` trong `build_graph.py`

2. **Slow Training**: Nếu training quá chậm:
   - Dùng GPU (CUDA)
   - Giảm `NUM_EPOCHS`
   - Giảm số samples
   - Dùng GCN thay vì GAT

3. **Poor Performance**: Nếu kết quả không tốt:
   - Thử model khác (GAT hoặc Hybrid)
   - Tăng `HIDDEN_CHANNELS`
   - Tăng `NUM_LAYERS`
   - Tăng `K_NEIGHBORS`
   - Thử điều chỉnh learning rate

## 🐛 Troubleshooting

### Error: "No module named 'torch_geometric'"

```bash
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Error: "CUDA out of memory"

Giảm batch size hoặc model size:
```python
HIDDEN_CHANNELS = 64
MAX_SAMPLES = 10000
```

### Error: "No CSV files found"

Kiểm tra đường dẫn `DATA_DIR` trong scripts

## 📚 References

- CICIDS2018 Dataset: https://www.unb.ca/cic/datasets/ids-2018.html
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Graph Neural Networks: Kipf & Welling (2017)
- Graph Attention Networks: Veličković et al. (2018)

## 📝 License

Educational purposes only.

## 👨‍💻 Author

Senior Data Engineer
Date: November 24, 2025

---

**Chúc bạn thành công với project! 🚀**


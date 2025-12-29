# GNN-IDS: Graph Neural Network for Intrusion Detection System

Hệ thống phát hiện xâm nhập mạng (Intrusion Detection System) sử dụng Graph Neural Networks với **kiến trúc modular** - 3 package độc lập cho từng nhiệm vụ.

---

## 📦 Kiến Trúc Modular

Dự án được tổ chức thành 3 package riêng biệt, mỗi package có config và logic riêng:

```
GRAPH_IOT/
├── main.py              # CLI chính để chạy các command
├── preprocess/          # Package xử lý dữ liệu
│   ├── config.yaml
│   ├── load_data.py
│   ├── preprocess.py
│   └── utils.py
├── flow_gnn/            # Package Flow-based GNN
│   ├── config.yaml
│   ├── graph.py         # KNN graph construction
│   ├── model.py         # FlowGraphSAGE model
│   ├── train.py         # Training logic
│   └── utils.py
└── endpoint_gnn/        # Package Endpoint-based GNN
    ├── config.yaml
    ├── graph.py         # Endpoint graph construction
    ├── model.py         # E-GraphSAGE model
    ├── train.py         # Training logic
    └── utils.py
```

### Ưu điểm của kiến trúc này:
✅ **Tách biệt rõ ràng**: Mỗi package độc lập, dễ maintain  
✅ **Config riêng**: Mỗi package có config.yaml riêng, không bị rối  
✅ **Dễ mở rộng**: Thêm package mới không ảnh hưởng code cũ  
✅ **Clean CLI**: Chỉ cần 1 main.py với commands đơn giản  

---

## 🎯 Hai Phương Pháp GNN

### 1. Flow-based GNN (flow_gnn/)
**Cách tiếp cận**: Node = flow record, Edge = KNN similarity

- **Node**: Mỗi flow record là 1 node
- **Edge**: KNN graph (k-nearest neighbors dựa trên cosine/euclidean distance)
- **Task**: Node classification (phân loại từng flow)
- **Model**: FlowGraphSAGE (GraphSAGE cho node classification)

**Ưu điểm**:
- Đơn giản, trực quan
- Không cần IP mapping phức tạp
- Phù hợp khi muốn phân loại từng flow độc lập

### 2. Endpoint-based GNN (endpoint_gnn/)
**Cách tiếp cận**: Node = endpoint (IP hoặc IP:Port), Edge = flow

- **Node**: Các endpoint (IP addresses hoặc IP:Port combinations)
- **Edge**: Flow records kết nối giữa các endpoints
- **Edge features**: Flow features (packet stats, duration, flags, etc.)
- **Task**: Edge classification (phân loại từng flow dựa trên context của endpoints)
- **Model**: E-GraphSAGE (Edge-feature-based GraphSAGE)

**Ưu điểm**:
- Tận dụng cấu trúc mạng thực tế
- Anti-leakage: IP random mapping tránh overfitting
- Phù hợp với bản chất của network traffic

---

## 🚀 Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd GRAPH_IOT

# Cài đặt dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- scikit-learn
- pandas
- numpy
- PyYAML

---

## 💻 Sử Dụng

### Command Line Interface

```bash
# Hiển thị help
python main.py --help

# Xem help cho từng command
python main.py flow --help
python main.py endpoint --help
python main.py preprocess --help
```

### 1. Train Flow-based GNN

```bash
python main.py flow --csv data/Tuesday_20_02_exist_ip.csv
```

**Options**:
```bash
python main.py flow \
  --csv data/Tuesday_20_02_exist_ip.csv \
  --config flow_gnn/config.yaml \
  --max-samples 200000 \
  --device auto
```

### 2. Train Endpoint-based GNN

```bash
python main.py endpoint --csv data/Tuesday_20_02_exist_ip.csv
```

**Options**:
```bash
python main.py endpoint \
  --csv data/Tuesday_20_02_exist_ip.csv \
  --config endpoint_gnn/config.yaml \
  --device auto
```

### 3. Preprocess Only

```bash
python main.py preprocess \
  --csv data/Tuesday_20_02_exist_ip.csv \
  --output output/preprocess/preprocessed.pt
```

---

## ⚙️ Cấu Hình

Mỗi package có file `config.yaml` riêng để cấu hình chi tiết.

### preprocess/config.yaml
```yaml
data:
  csv_path: "data/Tuesday_20_02_exist_ip.csv"
  max_samples: 200000
  test_split: 0.3
  val_split: 0.1

project:
  seed: 42
```

### flow_gnn/config.yaml
```yaml
model:
  hidden_dim: 128
  num_classes: 2
  num_layers: 2
  dropout: 0.3

graph:
  k_neighbors: 10
  metric: "cosine"

training:
  epochs: 100
  batch_size: 512
  learning_rate: 0.001
```

### endpoint_gnn/config.yaml
```yaml
model:
  hidden_dim: 128
  num_classes: 2
  num_layers: 2
  dropout: 0.2

graph:
  mapping_mode: "ip_port"  # hoặc "ip_only"
  anti_leakage:
    enabled: true
    map_scope: "all_ips"

training:
  epochs: 100
  batch_size: 1024
  learning_rate: 0.001
```

---

## 📊 Dataset

Sử dụng **CICIDS2018** - Network intrusion detection dataset

**Format**: CSV files với các cột:
- Flow features: Duration, packet counts, byte counts, flags, etc.
- Label: "Benign" hoặc attack types
- Network info: Src IP, Dst IP, Src Port, Dst Port

**Sample data location**: `data/Tuesday_20_02_exist_ip.csv`

---

## 🔬 Pipeline Chi Tiết

### Flow-based GNN Pipeline:
1. **Load CSV** → Parse và clean data
2. **Preprocess** → Split train/val/test, StandardScaler
3. **Build KNN Graph** → k-nearest neighbors based on feature similarity
4. **Create PyG Data** → Node features, edge_index, masks
5. **Train FlowGraphSAGE** → Mini-batch training với NeighborLoader
6. **Evaluate** → Accuracy, F1, Precision, Recall, FAR

### Endpoint-based GNN Pipeline:
1. **Load CSV** → Parse data
2. **Feature Engineering** → Extract numeric features, scale
3. **Build Endpoint Graph**:
   - Create endpoint nodes (IP or IP:Port)
   - Optional: Apply IP random mapping (anti-leakage)
   - Build edges from flow records
4. **Create PyG Data** → Node features (ones), edge features (flow), edge labels
5. **Train E-GraphSAGE** → Mini-batch edge classification
6. **Evaluate** → Edge-level metrics

---

## 📈 Metrics

Cả hai phương pháp đều đánh giá với các metrics:

- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Độ chính xác của attack predictions
- **Recall (Detection Rate)**: Tỷ lệ phát hiện attack
- **F1 Score**: Harmonic mean của Precision và Recall
- **FAR (False Alarm Rate)**: Tỷ lệ cảnh báo nhầm

---

## 📁 Output

Kết quả training được lưu trong thư mục tương ứng:

```
output/
├── flow_gnn/
│   └── best_model.pt
├── endpoint_gnn/
│   └── best_model.pt
└── preprocess/
    └── preprocessed.pt
```

---

## 🛠️ Development

### Thêm package mới:
1. Tạo thư mục mới (vd: `new_method/`)
2. Tạo `config.yaml` riêng
3. Implement model, graph, train, utils
4. Thêm command mới vào `main.py`

### Modify existing package:
- Chỉnh sửa code trong package tương ứng
- Update config.yaml nếu cần
- Không ảnh hưởng packages khác

---

## 📝 Examples

### Example 1: Quick test với sample nhỏ
```bash
python main.py flow --csv data/Tuesday_20_02_exist_ip.csv --max-samples 10000
```

### Example 2: Full training với custom config
```bash
# Chỉnh sửa flow_gnn/config.yaml trước
python main.py flow --config flow_gnn/config.yaml
```

### Example 3: So sánh hai phương pháp
```bash
# Train cả hai
python main.py flow --csv data/Tuesday_20_02_exist_ip.csv
python main.py endpoint --csv data/Tuesday_20_02_exist_ip.csv

# So sánh kết quả từ logs
```

---

## 🐛 Troubleshooting

**Out of Memory**:
- Giảm `max_samples` trong config
- Giảm `batch_size` trong config
- Sử dụng `--max-samples` flag

**MPS (Apple Silicon) issues**:
- Code tự động handle MPS compatibility
- Data kept on CPU for NeighborLoader, batches moved to MPS

**Import errors**:
- Đảm bảo chạy từ root directory: `python main.py ...`
- Check tất cả dependencies đã cài: `pip install -r requirements.txt`

---

## 📄 License

MIT License

---

## 👥 Authors

GNN-IDS Team

---

## 🙏 Acknowledgments

- CICIDS2018 Dataset
- PyTorch Geometric library
- GraphSAGE paper (Hamilton et al.)
- E-GraphSAGE approach

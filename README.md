# GNN-IDS: Graph Neural Network for Intrusion Detection System

Hệ thống phát hiện xâm nhập mạng (Intrusion Detection System) sử dụng Graph Neural Networks với hai phương pháp xây dựng đồ thị khác nhau để phân tích traffic mạng trên dataset CICIDS2018.

---

## 📋 Mục Lục
1. [Tổng Quan](#-tổng-quan)
2. [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
3. [Hai Phương Pháp Xây Dựng Đồ Thị](#-hai-phương-pháp-xây-dựng-đồ-thị)
4. [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
5. [Cấu Hình Tham Số](#️-cấu-hình-tham-số)
6. [Thuật Toán & Mô Hình](#-thuật-toán--mô-hình)
7. [Pipeline Xử Lý Dữ Liệu](#-pipeline-xử-lý-dữ-liệu)
8. [Training & Evaluation](#-training--evaluation)
9. [Metrics & Đánh Giá](#-metrics--đánh-giá)
10. [Cài Đặt & Sử Dụng](#-cài-đặt--sử-dụng)

---

## 🎯 Tổng Quan

### Mục Đích
Phát hiện các cuộc tấn mạng (attack) và phân loại traffic thành benign/attack bằng cách:
- Biểu diễn network flows dưới dạng đồ thị
- Sử dụng Graph Neural Networks để học representation
- Phân loại flows/endpoints dựa trên đặc trưng và cấu trúc đồ thị

### Đặc Điểm Chính
- ✅ **Hai chế độ**: Flow-based (Node Classification) & IP-based (Edge Classification)
- ✅ **Anti-leakage**: IP randomization để tránh overfitting trên địa chỉ IP
- ✅ **Class imbalance**: Weighted loss function xử lý mất cân bằng dữ liệu
- ✅ **Mini-batch training**: Neighbor sampling cho scalability
- ✅ **Early stopping**: Tự động dừng khi không cải thiện
- ✅ **Comprehensive metrics**: Accuracy, F1, Precision, Recall, FAR, Detection Rate

### Dataset
- **CICIDS2018**: Network intrusion detection dataset
- **Format**: CSV files chứa flow records
- **Features**: 80+ features (packet statistics, duration, flags, etc.)
- **Labels**: Binary (benign/attack) hoặc multiclass (attack types)

---

## 🏗️ Kiến Trúc Hệ Thống

```
Input: CSV Files (Network Flows)
    ↓
Data Loading & Preprocessing
    ↓
Graph Construction ← [Two Modes]
    ↓
GNN Model (GraphSAGE variants)
    ↓
Classification (Node/Edge)
    ↓
Output: Predictions & Metrics
```

### Workflow Tổng Quát
1. **Load data** từ CSV files (CICIDS2018)
2. **Preprocess**: Drop columns, handle inf/nan, scaling
3. **Build graph**: KNN hoặc Endpoint-based
4. **Train GNN**: GraphSAGE hoặc E-GraphSAGE
5. **Evaluate**: Compute metrics trên test set
6. **Save**: Checkpoints và predictions

---

## 📊 Hai Phương Pháp Xây Dựng Đồ Thị

### 1️⃣ Flow-based Mode (Node Classification)

**Cấu trúc đồ thị:**
- **Nodes**: Mỗi node = 1 flow record
- **Edges**: KNN graph dựa trên độ tương đồng features
- **Task**: Classify mỗi node (flow) là benign hay attack

**Cách hoạt động:**
```python
# Build KNN graph
adj = kneighbors_graph(X_scaled, k=10, metric='cosine')
# Nodes share edge if they are k-nearest neighbors
```

**Ưu điểm:**
- Đơn giản, dễ hiểu
- Phù hợp với flow-level analysis
- KNN graph capture được feature similarity

**Nhược điểm:**
- Không model được network topology thực tế
- KNN expensive với large dataset

**Model**: FlowGraphSAGE (standard GraphSAGE cho node classification)

---

### 2️⃣ Endpoint-based Mode (Edge Classification) - E-GraphSAGE

**Cấu trúc đồ thị:**
- **Nodes**: Endpoints (IP:Port combinations hoặc chỉ IP)
- **Edges**: Mỗi edge = 1 flow record giữa 2 endpoints
- **Edge features**: Flow features (packet stats, duration, etc.)
- **Task**: Classify mỗi edge (flow) là benign hay attack

**Cách hoạt động:**
```python
# Node = IP:Port
src_endpoint = f"{src_ip}:{src_port}"
dst_endpoint = f"{dst_ip}:{dst_port}"

# Edge = flow giữa src → dst
edge = (src_endpoint, dst_endpoint)
edge_features = flow_features  # 78 features
edge_label = benign/attack
```

**Ưu điểm:**
- Model được network topology thực tế
- Node embeddings học được behavior của endpoints
- Anti-leakage: IP randomization ngăn overfitting
- Scalable với LinkNeighborLoader

**Nhược điểm:**
- Phức tạp hơn flow-based
- Cần nhiều memory hơn

**Model**: E-GraphSAGE (GraphSAGE variant cho edge classification)

**Anti-leakage Mechanism:**
```yaml
anti_leakage:
  enabled: true
  map_scope: "all_ips"  # hoặc "src_ip_only"
  
# Ánh xạ IP → random ID
# VD: 192.168.1.1 → IP_000001
#     10.0.0.5   → IP_000002
# Tránh model học thuộc địa chỉ IP thay vì pattern
```

---

## 📁 Cấu Trúc Dự Án

```
GRAPH_IOT/
│
├── config.yaml                 # ⚙️ File cấu hình chính (tất cả parameters)
├── requirements.txt            # 📦 Dependencies
├── README.md                   # 📖 Documentation
│
├── data/                       # 📊 Dataset CSV files
│   ├── Friday_02_03.csv
│   ├── Tuesday_20_02_exist_ip.csv
│   └── ...
│
├── notebook/                   # 📓 Jupyter notebooks
│   └── kaggle_gnn_ids.ipynb   # Experiments & visualization
│
└── src/                        # 💻 Source code
    ├── main.py                # Entry point cho Flow-based mode
    ├── main_v2.py             # Entry point tổng quát (cả 2 modes)
    │
    └── gnn_ids/               # Main package
        │
        ├── config.py          # Load config từ YAML
        ├── config_loader.py   # Config utilities
        ├── common.py          # Set seed, device selection
        │
        ├── data/              # 📥 Data processing
        │   ├── load_csv.py           # Load & clean CSV
        │   ├── preprocess.py         # Split & scaling
        │   └── graph_build.py        # Endpoint graph construction
        │
        ├── graph/             # 🕸️ Graph construction
        │   ├── knn_graph.py          # KNN graph builder
        │   └── sage.py               # (if any graph utils)
        │
        ├── model/             # 🧠 GNN Models
        │   ├── sage.py               # FlowGraphSAGE (node classification)
        │   └── e_graphsage.py        # E-GraphSAGE (edge classification)
        │
        ├── train.py           # 🏋️ Training loop (Flow-based)
        ├── train_edge.py      # 🏋️ Training loop (Endpoint-based)
        ├── eval.py            # 📊 Evaluation utilities
        │
        └── utils/             # 🛠️ Utilities
            └── metrics.py            # FAR, DR, F1, etc.
```

### Vai Trò Các Module

**Data Processing:**
- `load_csv.py`: Load CSV, drop columns, handle inf/nan
- `preprocess.py`: Train/val/test split, StandardScaler/MinMaxScaler
- `graph_build.py`: Build endpoint mapping, create edge_index

**Graph Construction:**
- `knn_graph.py`: KNN graph với cosine/euclidean distance
- Symmetrize để tạo undirected graph

**Models:**
- `sage.py`: FlowGraphSAGE - standard SAGEConv layers cho node classification
- `e_graphsage.py`: E-GraphSAGE - EdgeFeatureSAGEConv + edge classifier

**Training:**
- `train.py`: Training loop với NeighborLoader (node classification)
- `train_edge.py`: Training loop với LinkNeighborLoader (edge classification)
- EarlyStopping, checkpointing, logging

**Evaluation:**
- `metrics.py`: Compute FAR, Detection Rate, F1, confusion matrix
- `eval.py`: Test evaluation, save predictions

---

## ⚙️ Cấu Hình Tham Số

File `config.yaml` chứa tất cả parameters:

### Project Settings
```yaml
project:
  name: "GNN-IDS"
  seed: 42               # Random seed cho reproducibility
  device: "auto"         # auto/cuda/mps/cpu - tự động detect GPU
```

### Data Configuration
```yaml
data:
  csv_path: "data/Tuesday_20_02_exist_ip.csv"
  label_col: "Label"     # Cột label
  
  # Endpoint columns
  src_ip_col: "Src IP"
  src_port_col: "Src Port"
  dst_ip_col: "Dst IP"
  dst_port_col: "Dst Port"
  
  # Columns to drop (non-numeric/metadata)
  drop_cols:
    - "Flow ID"
    - "Timestamp"
    - "Src IP"
    - "Dst IP"
    - "Src Port"
    - "Dst Port"
  
  max_samples: 20000000  # Giới hạn số samples (memory constraint)
  test_split: 0.3        # 30% test set
  val_split: 0.1         # 10% validation set
```

**Giải thích:**
- `drop_cols`: Loại bỏ metadata và non-numeric columns trước khi training
- `max_samples`: Limit để tránh OOM với large datasets
- Splits: 60% train, 10% val, 30% test

### Mode Selection
```yaml
mode: "endpoint"  # "flow" hoặc "endpoint"
```

### Flow-based Graph Config
```yaml
flow_graph:
  k_neighbors: 10        # K cho KNN graph
  metric: "cosine"       # cosine/euclidean/manhattan
```

**KNN Graph:**
- Mỗi node connect đến k nearest neighbors dựa trên feature similarity
- Cosine metric: tốt cho high-dimensional sparse data
- Graph được symmetrize: nếu A → B thì B → A

### Endpoint-based Graph Config
```yaml
endpoint_graph:
  mapping_mode: "ip_port"  # ip_port hoặc ip_only
  
  anti_leakage:
    enabled: false          # Enable IP randomization
    map_scope: "all_ips"    # all_ips hoặc src_ip_only
```

**Mapping modes:**
- `ip_port`: Node = IP:Port (chi tiết hơn, nhiều nodes)
- `ip_only`: Node = IP (ít nodes hơn, aggregate flows)

**Anti-leakage:**
- Ánh xạ IP addresses sang random IDs
- Ngăn model memorize specific IPs
- `all_ips`: map cả src và dst IPs
- `src_ip_only`: chỉ map source IPs

### Preprocessing
```yaml
preprocessing:
  scale: "standard"      # standard hoặc minmax
  handle_inf: "zero"     # zero hoặc median
```

**Scaling:**
- `standard`: StandardScaler (zero mean, unit variance)
- `minmax`: MinMaxScaler (scale to [0, 1])

**Handle inf/nan:**
- Replace inf với 0 hoặc median của column

### Task Configuration
```yaml
task:
  type: "binary"         # binary hoặc multiclass
  num_classes: 2         # 2 cho binary, >2 cho multiclass
```

### Model: FlowGraphSAGE
```yaml
flow_model:
  type: "GraphSAGE"
  hidden_dim: 128        # Hidden layer dimension
  num_layers: 2          # Số GraphSAGE layers
  dropout: 0.3           # Dropout rate
  aggregator: "mean"     # mean/max/lstm
```

**Architecture:**
```
Input (78 features)
  ↓
SAGEConv(78 → 128) + BatchNorm + ReLU + Dropout
  ↓
SAGEConv(128 → 128) + BatchNorm + ReLU + Dropout
  ↓
Linear(128 → 2) → Logits
```

### Model: E-GraphSAGE
```yaml
endpoint_model:
  type: "E-GraphSAGE"
  hidden_dim: 128
  num_layers: 2          # K=2 theo paper
  dropout: 0.2
  aggregator: "mean"
  activation: "relu"
```

**Architecture:**
```
Input: Node features (initialized as zeros)
       Edge features (78 flow features)

EdgeFeatureSAGEConv layers (aggregate edge features)
  ↓
Node embeddings (128-dim)
  ↓
Edge representation: concat(z_src, z_dst) → 256-dim
  ↓
Edge Classifier:
  Linear(256 → 128) + ReLU + Dropout
  Linear(128 → 2) → Logits
```

**EdgeFeatureSAGEConv:**
- Standard SAGEConv aggregate neighbor nodes
- EdgeFeatureSAGEConv aggregate edge features của incoming edges
- Update node embedding = f(self features, aggregated edge features)

### Training Configuration
```yaml
training:
  epochs: 50
  batch_size: 4096           # Cho LinkNeighborLoader
  learning_rate: 0.001
  weight_decay: 0.0001       # L2 regularization
  optimizer: "adam"
  
  num_neighbors: [15, 10]    # 2-hop neighbor sampling
  
  early_stopping:
    enabled: true
    patience: 10             # Chờ 10 epochs
    min_delta: 0.0001        # Min improvement
    metric: "f1"             # f1/accuracy/loss
  
  loss:
    type: "cross_entropy"
    use_class_weights: true  # Handle class imbalance
```

**Neighbor Sampling:**
- `[15, 10]`: Sample 15 neighbors ở hop 1, 10 neighbors ở hop 2
- Giảm computation graph size
- Tradeoff: speed vs accuracy

**Class Weights:**
```python
# Compute from label distribution
class_weights = n_samples / (n_classes * np.bincount(labels))
# Example: {0: 0.52, 1: 2.3} nếu class 1 minority
criterion = CrossEntropyLoss(weight=class_weights)
```

**Early Stopping:**
- Monitor F1 score trên validation set
- Stop nếu không improve sau 10 epochs
- Save best model checkpoint

---

## 🧠 Thuật Toán & Mô Hình

### GraphSAGE (SAmple and aggreGatE)

**Ý tưởng:**
- Thay vì propagate thông tin từ toàn bộ neighbors (như GCN)
- Sample một subset neighbors và aggregate

**Algorithm:**
```
For each layer l:
  1. Sample neighbors: N(v) → sampled N'(v)
  2. Aggregate: h_N(v) = AGGREGATE({h_u^(l-1) : u ∈ N'(v)})
  3. Update: h_v^l = σ(W · CONCAT(h_v^(l-1), h_N(v)))
  4. Normalize: h_v^l = h_v^l / ||h_v^l||
```

**Aggregator types:**
- **Mean**: average của neighbor embeddings
- **Max**: element-wise max
- **LSTM**: sequential aggregation

**Ưu điểm:**
- Scalable: không cần load full graph
- Inductive: có thể inference trên unseen nodes
- Flexible: support nhiều aggregator functions

### FlowGraphSAGE (Node Classification)

**Implementation:**
```python
class FlowGraphSAGE(nn.Module):
    def __init__(self, in_dim=78, hidden_dim=128, num_classes=2):
        self.convs = [
            SAGEConv(in_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim)
        ]
        self.bns = [BatchNorm1d(hidden_dim), ...]
        self.classifier = Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = relu(x)
            x = dropout(x)
        return self.classifier(x)
```

**Training:**
- NeighborLoader sample subgraphs
- Mini-batch training
- Loss chỉ compute trên seed nodes (batch center nodes)

### E-GraphSAGE (Edge Classification)

**Khác biệt với standard GraphSAGE:**
1. **EdgeFeatureSAGEConv**: Aggregate edge features thay vì node features
2. **Edge Classifier**: Classify edges thay vì nodes

**EdgeFeatureSAGEConv:**
```python
class EdgeFeatureSAGEConv(nn.Module):
    def forward(self, x, edge_index, edge_attr):
        # 1. Aggregate edge features for each node
        aggregated = scatter_add(edge_attr, dst_index, dim=0)
        aggregated = aggregated / degree  # Mean aggregation
        
        # 2. Combine with self features
        out = W_self @ x + W_neigh @ aggregated + bias
        return out
```

**Edge Representation:**
```python
# Get node embeddings z_u, z_v
z_u = node_embeddings[src]
z_v = node_embeddings[dst]

# Edge embedding = concatenate
edge_emb = concat([z_u, z_v])  # Shape: [num_edges, 2*hidden_dim]

# Classify
logits = edge_classifier(edge_emb)
```

**Training với LinkNeighborLoader:**
```python
loader = LinkNeighborLoader(
    data,
    num_neighbors=[15, 10],
    edge_label_index=edge_label_index,  # Edges to classify
    batch_size=4096
)

for batch in loader:
    logits = model(batch.x, batch.edge_index, batch.edge_attr,
                   edge_label_index=batch.edge_label_index)
    loss = criterion(logits, batch.edge_label)
```

**Message Passing:**
```
Round 1:
  - Each node aggregates features từ incident edges (distance=1)
  - Update node embedding h_v^(1)

Round 2:
  - Each node aggregates từ neighbors' edges (distance=2)
  - Update node embedding h_v^(2)

Edge Classification:
  - Edge (u,v) → concat(h_u^(2), h_v^(2))
  - MLP classifier → benign/attack
```

---

## 🔄 Pipeline Xử Lý Dữ Liệu

### 1. Load CSV
```python
# load_csv.py
df = pd.read_csv(csv_path)

# Basic cleaning
df = df.drop(columns=drop_cols)  # Drop metadata
df = df.replace([np.inf, -np.inf], np.nan)  # Handle inf
df = df.fillna(0)  # or fillna(median)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[label_col])
X = df.drop(columns=[label_col])
```

### 2. Preprocess & Split
```python
# preprocess.py
# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=seed, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.75, stratify=y_temp
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# To tensor
x_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)
```

### 3A. Build KNN Graph (Flow-based)
```python
# knn_graph.py
from sklearn.neighbors import kneighbors_graph

# Build KNN graph
adj = kneighbors_graph(
    X_scaled, 
    n_neighbors=k, 
    metric='cosine',
    mode='distance'
)

# Symmetrize
adj = adj.maximum(adj.T)

# Extract edges
row, col = adj.nonzero()
edge_index = torch.tensor([row, col], dtype=torch.long)

# Edge weights = similarity
distances = adj[row, col]
edge_weight = 1.0 - distances  # Convert distance to similarity
```

### 3B. Build Endpoint Graph (Endpoint-based)
```python
# graph_build.py
# Build endpoint mapping
endpoint_to_idx = {}
for idx, (src_ip, src_port, dst_ip, dst_port) in enumerate(flows):
    src_ep = f"{src_ip}:{src_port}"
    dst_ep = f"{dst_ip}:{dst_port}"
    
    if src_ep not in endpoint_to_idx:
        endpoint_to_idx[src_ep] = len(endpoint_to_idx)
    if dst_ep not in endpoint_to_idx:
        endpoint_to_idx[dst_ep] = len(endpoint_to_idx)

# Create edge_index
src_indices = [endpoint_to_idx[src_ep] for src_ep in src_endpoints]
dst_indices = [endpoint_to_idx[dst_ep] for dst_ep in dst_endpoints]
edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

# Edge features = flow features
edge_attr = torch.tensor(X_scaled, dtype=torch.float)

# Edge labels
edge_label = torch.tensor(y, dtype=torch.long)
```

### 4. Create PyG Data Object
```python
from torch_geometric.data import Data

# Flow-based
data = Data(
    x=x_tensor,              # Node features [N, 78]
    edge_index=edge_index,   # Edges [2, E]
    y=y_tensor,              # Node labels [N]
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

# Endpoint-based
data = Data(
    x=node_features,         # Node features [num_endpoints, dim]
    edge_index=edge_index,   # [2, num_flows]
    edge_attr=edge_attr,     # Edge features [num_flows, 78]
    edge_label=edge_label,   # Edge labels [num_flows]
    train_edge_mask=train_mask,
    val_edge_mask=val_mask,
    test_edge_mask=test_mask
)
```

---

## 🏋️ Training & Evaluation

### Training Loop (Endpoint-based)

```python
# train_edge.py
def train_epoch_edge(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward
        logits = model(
            batch.x, 
            batch.edge_index,
            batch.edge_attr,
            edge_label_index=batch.edge_label_index
        )
        
        # Loss
        loss = criterion(logits, batch.edge_label)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

### Evaluation
```python
# eval.py
@torch.no_grad()
def evaluate_edge(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr,
                      edge_label_index=batch.edge_label_index)
        preds = logits.argmax(dim=-1)
        
        all_preds.append(preds.cpu())
        all_labels.append(batch.edge_label.cpu())
    
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    
    metrics = compute_metrics(y_true, y_pred)
    return metrics
```

### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, metric='f1'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_score = None
        self.counter = 0
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
```

### Full Training Pipeline
```python
# main_v2.py
def main():
    # 1. Load data
    X, y = load_data(csv_path)
    
    # 2. Preprocess
    X_scaled, y_tensor, splits = preprocess(X, y)
    
    # 3. Build graph
    if mode == "flow":
        data = build_knn_graph(X_scaled, y_tensor, k=10)
    else:
        data = build_endpoint_graph(df, X_scaled, y_tensor)
    
    # 4. Create loaders
    train_loader = LinkNeighborLoader(data, ...)
    val_loader = LinkNeighborLoader(data, ...)
    test_loader = LinkNeighborLoader(data, ...)
    
    # 5. Initialize model
    model = EGraphSAGE(in_dim=78, hidden_dim=128, num_classes=2)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Class weights for imbalance
    class_weights = compute_class_weights(y_train)
    criterion = CrossEntropyLoss(weight=class_weights)
    
    # Early stopping
    early_stop = EarlyStopping(patience=10, metric='f1')
    
    # 6. Training loop
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, ...)
        val_metrics = evaluate(model, val_loader, ...)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, "
              f"Val F1={val_metrics['f1']:.4f}")
        
        # Early stopping
        if early_stop(val_metrics['f1']):
            print("Early stopping triggered")
            break
        
        # Save checkpoint
        if val_metrics['f1'] > best_f1:
            save_checkpoint(model, optimizer, epoch, val_metrics)
            best_f1 = val_metrics['f1']
    
    # 7. Test evaluation
    test_metrics = evaluate(model, test_loader, ...)
    print(f"Test Results: {test_metrics}")
    
    # 8. Save predictions
    save_predictions(y_pred, y_true, output_path)
```

---

## 📊 Metrics & Đánh Giá

### Metrics Computation
```python
# utils/metrics.py
def compute_metrics(y_true, y_pred, task_type='binary'):
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    if task_type == 'binary':
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Precision, Recall, F1
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1'] = f1_score(y_true, y_pred)
        
        # IDS-specific metrics
        metrics['far'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['detection_rate'] = metrics['recall']
        
    return metrics
```

### Metrics Giải Thích

**1. Accuracy**: Tỷ lệ dự đoán đúng tổng thể
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**2. Precision**: Trong các mẫu dự đoán là attack, bao nhiêu thực sự là attack
```
Precision = TP / (TP + FP)
```

**3. Recall (Detection Rate)**: Trong các attack thực tế, phát hiện được bao nhiêu
```
Recall = TP / (TP + FN)
```

**4. F1 Score**: Harmonic mean của Precision và Recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**5. False Alarm Rate (FAR)**: Tỷ lệ benign flows bị classify nhầm thành attack
```
FAR = FP / (FP + TN)
```

**Confusion Matrix:**
```
                 Predicted
                 0       1
Actual  0       TN      FP
        1       FN      TP
```
- TN: True Negative (benign predicted as benign)
- FP: False Positive (benign predicted as attack) - False alarm
- FN: False Negative (attack predicted as benign) - Missed detection
- TP: True Positive (attack predicted as attack) - Correct detection

**Trade-offs:**
- High Recall, Low Precision: Nhiều false alarms
- High Precision, Low Recall: Miss nhiều attacks
- F1 score: Balance giữa hai metrics

**Ideal IDS:**
- High Detection Rate (Recall > 95%)
- Low FAR (< 5%)
- High F1 score (> 90%)

---

## 🚀 Cài Đặt & Sử Dụng

### 1. Cài Đặt Dependencies

```bash
# Clone repository
git clone <repo_url>
cd GRAPH_IOT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

```bash
python src/main_v2.py --config config.yaml --mode endpoint
```

#### Flow Mode (KNN-based)

```bash
python src/main_v2.py --config config.yaml --mode flow
```

#### With custom CSV path

```bash
python src/main_v2.py --csv_path path/to/data.csv --mode endpoint
```

## 📊 E-GraphSAGE Architecture

### Graph Structure
- **Nodes**: Endpoints (IP:Port)
- **Edges**: Flow records
- **Node features**: All-ones vector (dimension = num_edge_features)
- **Edge features**: Flow statistics (scaled)
- **Edge labels**: Binary (benign/attack) or multiclass

### Model Details

```
Input: 
  - Node features: [num_nodes, F] (all ones)
  - Edge features: [num_edges, F]
  - Edge index: [2, num_edges]

Layer 1: EdgeFeatureSAGEConv
  - Aggregate edge features for each node
  - Transform: Linear(F → 128)
  - Activation: ReLU
  - Dropout: 0.2

Layer 2: EdgeFeatureSAGEConv
  - Aggregate: Linear(128 → 128)
  - Activation: ReLU
  - Dropout: 0.2

Edge Embedding:
  - For edge (u,v): concat(z_u, z_v) → [256]

Edge Classifier:
  - Linear(256 → 128) → ReLU → Dropout
  - Linear(128 → num_classes)
```

### Key Innovation (E-GraphSAGE)
- Message passing uses **edge features** instead of only node features
- Each node aggregates features from **incident edges**
- Edge prediction via concatenated node embeddings

## ⚙️ Configuration Reference

### Complete Configuration Template

The `config.yaml` file controls all aspects of the system. Here's a complete template with all available options:

```yaml
# =============================================================================
# GNN-IDS Configuration
# =============================================================================

# Project Settings
project:
  name: "GNN-IDS"
  seed: 42
  device: "auto"  # auto, cuda, mps, cpu

# Data Configuration
data:
  csv_path: "data/Tuesday_20_02.csv"
  label_col: "Label"
  
  # Column mapping for endpoint-based approach
  src_ip_col: "Src IP"
  src_port_col: "Src Port"
  dst_ip_col: "Dst IP"
  dst_port_col: "Dst Port"
  
  # Columns to drop (non-numeric/irrelevant)
  drop_cols:
    - "Flow ID"
    - "Timestamp"
    - "Src IP"
    - "Dst IP"
    - "Src Port"
    - "Dst Port"
  
  max_samples: 100000  # Limit for memory
  test_split: 0.3
  val_split: 0.1

# Graph Mode Selection
mode: "endpoint"  # "flow" (KNN) or "endpoint" (E-GraphSAGE)

# Flow-based Graph (node=flow, edge=KNN)
flow_graph:
  k_neighbors: 10
  metric: "cosine"  # cosine, euclidean, manhattan

# Endpoint-based Graph (node=endpoint, edge=flow)
endpoint_graph:
  mapping_mode: "ip_port"  # ip_port or ip_only
  
  # Anti-leakage settings
  anti_leakage:
    enabled: false  # Enable IP random mapping
    map_scope: "all_ips"  # all_ips or src_ip_only

# Preprocessing
preprocessing:
  scale: "standard"  # standard or minmax
  handle_inf: "zero"  # zero or median

# Task Configuration
task:
  type: "binary"  # binary or multiclass
  num_classes: 2
  class_weight: "balanced"  # balanced or null

# Model: Flow-based GraphSAGE (node classification)
flow_model:
  type: "GraphSAGE"
  hidden_dim: 128
  num_layers: 2
  dropout: 0.3
  aggregator: "mean"

# Model: E-GraphSAGE (edge classification)
endpoint_model:
  type: "E-GraphSAGE"
  hidden_dim: 128
  num_layers: 2  # K=2 as per paper
  dropout: 0.2
  aggregator: "mean"
  activation: "relu"
  
# Training Configuration
training:
  epochs: 30
  batch_size: 4096  # For LinkNeighborLoader
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"
  
  # Neighbor sampling
  num_neighbors: [15, 10]  # 2-hop sampling: [hop1, hop2]
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.0001
    metric: "f1"  # f1, accuracy, loss
  
  # Loss function
  loss:
    type: "cross_entropy"
    use_class_weights: true

# Evaluation
evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"
    - "far"  # False Alarm Rate
    - "confusion_matrix"
  
  save_predictions: true

# Output Configuration
output:
  dir: "src/output"
  checkpoint_dir: "src/output/checkpoints"
  log_dir: "src/output/logs"
  save_best_model: true

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Quick Configuration Examples

#### Example 1: Endpoint Mode (Recommended)

```yaml
mode: "endpoint"

data:
  csv_path: "data/CICIDS2018.csv"
  max_samples: 100000
  test_split: 0.3
  val_split: 0.1

endpoint_graph:
  mapping_mode: "ip_port"
  anti_leakage:
    enabled: false

endpoint_model:
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2

training:
  epochs: 30
  batch_size: 4096
  learning_rate: 0.001
  num_neighbors: [15, 10]
```

#### Example 2: Flow Mode (KNN-based)

```yaml
mode: "flow"

data:
  csv_path: "data/CICIDS2018.csv"
  max_samples: 100000

flow_graph:
  k_neighbors: 10
  metric: "cosine"

flow_model:
  hidden_dim: 128
  num_layers: 2
  dropout: 0.3

training:
  epochs: 50
  batch_size: 1024
  learning_rate: 0.001
  num_neighbors: [10, 5]
```

#### Example 3: Anti-Leakage Mode

```yaml
mode: "endpoint"

endpoint_graph:
  mapping_mode: "ip_port"
  anti_leakage:
    enabled: true        # Enable IP randomization
    map_scope: "all_ips" # Randomize all IPs

# ... rest of config
```

### Configuration Parameter Explanations

#### Device Selection
- `auto`: Automatically selects CUDA > MPS > CPU
- `cuda`: Force NVIDIA GPU (if available)
- `mps`: Force Apple Metal (for M1/M2 Macs)
- `cpu`: Force CPU (for testing/debugging)

#### Neighbor Sampling
- `num_neighbors: [15, 10]`: Sample 15 neighbors at hop 1, 10 at hop 2
- Higher values = more context but slower training
- Lower values = faster but less information

#### Early Stopping
- `patience`: Number of epochs without improvement before stopping
- `min_delta`: Minimum change to count as improvement
- `metric`: Which metric to monitor (f1, accuracy, loss)

#### Anti-Leakage
- Prevents model from memorizing specific IP addresses
- `map_scope: "all_ips"`: Randomize all IP addresses
- `map_scope: "src_ip_only"`: Only randomize source IPs

## 📈 Evaluation Metrics

The framework computes:
- **Accuracy**: Overall correctness
- **Precision**: Attack detection precision
- **Recall/Detection Rate (DR)**: Attack detection recall
- **F1 Score**: Harmonic mean of precision & recall
- **False Alarm Rate (FAR)**: FP / (FP + TN)
- **Confusion Matrix**: Detailed classification breakdown

### Example Output

```
Test Metrics:
--------------------------------------------------
accuracy            : 0.9823
precision           : 0.9756
recall              : 0.9891
f1                  : 0.9823
detection_rate      : 0.9891
far                 : 0.0145
--------------------------------------------------

Test Classification Report:
              precision    recall  f1-score   support

      Benign       0.99      0.98      0.98     15234
      Attack       0.98      0.99      0.98     14766

    accuracy                           0.98     30000

Confusion Matrix:
[[14912   322]
 [  161 14605]]
```

## 🔬 Anti-Leakage Features

### IP Random Mapping

Prevents the model from memorizing specific IP addresses:

```yaml
endpoint_graph:
  anti_leakage:
    enabled: true
    map_scope: "all_ips"  # Randomize all IPs
```

- Maps IP addresses to random tokens
- Preserves graph structure
- Reduces overfitting to specific IPs
- Improves generalization

## 💡 Usage Tips

### 1. Memory Management
- Reduce `max_samples` if OOM
- Decrease `batch_size` 
- Lower `num_neighbors` sampling

### 2. Class Imbalance
- Enable `use_class_weights: true`
- Monitor FAR and Detection Rate
- Consider SMOTE for severe imbalance

### 3. Hyperparameter Tuning
Key parameters:
- `hidden_dim`: 64, 128, 256
- `num_layers`: 2, 3
- `dropout`: 0.1-0.3
- `learning_rate`: 1e-4, 1e-3, 1e-2
- `num_neighbors`: [10, 5], [15, 10], [20, 15]

### 4. Dataset Requirements

CSV must contain:
- Numeric feature columns
- Label column (e.g., "Label")
- For endpoint mode: IP and Port columns

Supported datasets:
- CICIDS2017
- CICIDS2018
- CSE-CIC-IDS2018
- Any network flow CSV with similar structure

## 🔍 Comparison: Flow vs Endpoint Mode

| Aspect | Flow Mode | Endpoint Mode |
|--------|-----------|---------------|
| **Graph** | KNN (similarity) | Flow connections |
| **Nodes** | Flow records | Endpoints (IP:Port) |
| **Edges** | k-nearest neighbors | Flow records |
| **Task** | Node classification | Edge classification |
| **Model** | FlowGraphSAGE | E-GraphSAGE |
| **Features** | Node = flow features | Edge = flow features |
| **Pros** | Simple, interpretable | Realistic network structure |
| **Cons** | Artificial edges | More complex |

## 🐛 Troubleshooting

### Out of Memory
```yaml
data:
  max_samples: 50000  # Reduce dataset size

training:
  batch_size: 2048    # Smaller batches
  num_neighbors: [10, 5]  # Less sampling
```

### Poor Performance
- Enable class weights
- Increase epochs
- Try different learning rates
- Check data quality (inf/nan values)
- Verify label distribution

### CUDA Errors
```yaml
project:
  device: "cpu"  # Force CPU if GPU issues
```

## 📚 References

### Papers
- Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- E-GraphSAGE: "Edge-based Graph SAGE for Network Traffic Analysis"

### Datasets
- [CICIDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- [CSE-CIC-IDS2018](https://registry.opendata.aws/cse-cic-ids2018/)

```

**PyTorch Geometric:**
- torch-geometric >= 2.3.0
- torch-scatter, torch-sparse, torch-cluster

### 2. Prepare Data

Download CICIDS2018 dataset và đặt vào thư mục `data/`:
```bash
# Đảm bảo CSV files có các columns:
# - Src IP, Src Port, Dst IP, Dst Port
# - Flow features (80+ columns)
# - Label (benign/attack types)
```

### 3. Configure

Chỉnh sửa `config.yaml`:
```yaml
# Chọn mode
mode: "endpoint"  # hoặc "flow"

# Data path
data:
  csv_path: "data/Tuesday_20_02_exist_ip.csv"
  max_samples: 100000  # Tùy chỉnh theo RAM

# Model settings
endpoint_model:
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2

# Training settings  
training:
  epochs: 50
  batch_size: 4096
  learning_rate: 0.001
```

### 4. Training

#### Flow-based Mode
```bash
python src/main.py --csv_path data/Tuesday_20_02_exist_ip.csv
```

#### Endpoint-based Mode (Recommended)
```bash
python src/main_v2.py
```

**Training sẽ:**
1. Load và preprocess data
2. Build graph (KNN hoặc Endpoint-based)
3. Initialize model
4. Train với mini-batches
5. Evaluate trên validation set mỗi epoch
6. Save best checkpoint
7. Test evaluation và save predictions

**Output:**
```
output/
├── checkpoints/
│   └── best_model.pt          # Best model checkpoint
├── logs/
│   └── training.log           # Training logs
└── predictions.csv            # Test predictions
```

### 5. Monitor Training

```
Epoch 1/50: 100%|████████| 250/250 [00:15<00:00, 16.3it/s]
  Train Loss: 0.2845  Train Acc: 89.34%
  Val Loss: 0.1923    Val Acc: 93.21%
  Val F1: 0.9187      Val FAR: 0.0432
  
Epoch 2/50: 100%|████████| 250/250 [00:14<00:00, 17.1it/s]
  Train Loss: 0.1654  Train Acc: 94.12%
  Val Loss: 0.1456    Val Acc: 95.67%
  Val F1: 0.9523      Val FAR: 0.0289
  ✓ New best F1: 0.9523 (saved checkpoint)
  
...

Early stopping triggered after 10 epochs without improvement
Best Val F1: 0.9712 at epoch 18

Testing...
Test Results:
  Accuracy:  96.45%
  Precision: 0.9534
  Recall:    0.9812
  F1 Score:  0.9671
  FAR:       0.0198
  DR:        0.9812
```

### 6. Predictions

File `output/predictions.csv`:
```csv
true_label,predicted_label,probability_0,probability_1
0,0,0.9823,0.0177
1,1,0.0234,0.9766
0,0,0.9456,0.0544
1,1,0.1123,0.8877
...
```

---

## 🔧 Advanced Usage

### Custom Configuration

Tạo custom config file:
```bash
cp config.yaml config_custom.yaml
# Edit config_custom.yaml
python src/main_v2.py --config config_custom.yaml
```

### Hyperparameter Tuning

Modify `config.yaml`:
```yaml
endpoint_model:
  hidden_dim: [64, 128, 256]  # Test different sizes
  num_layers: [2, 3]
  dropout: [0.1, 0.2, 0.3]

training:
  learning_rate: [0.001, 0.0001]
  batch_size: [2048, 4096, 8192]
```

### Resume Training

```python
# Load checkpoint và continue
checkpoint = torch.load('output/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Inference Only

```python
# Load trained model
model = EGraphSAGE(in_dim=78, hidden_dim=128, num_classes=2)
checkpoint = torch.load('output/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict on new data
with torch.no_grad():
    logits = model(data.x, data.edge_index, data.edge_attr, 
                   edge_label_index=test_edges)
    predictions = logits.argmax(dim=-1)
```

---

## 📈 Expected Results

### Flow-based Mode (KNN Graph)
```
Metrics:
  Accuracy:  92-95%
  F1 Score:  0.89-0.93
  FAR:       3-7%
  DR:        91-96%

Training Time: ~5-10 min (100K samples)
Memory: ~4-6 GB
```

### Endpoint-based Mode (E-GraphSAGE)
```
Metrics:
  Accuracy:  95-98%
  F1 Score:  0.94-0.97
  FAR:       1-3%
  DR:        95-99%

Training Time: ~10-20 min (100K samples)
Memory: ~6-10 GB
```

**E-GraphSAGE thường tốt hơn vì:**
- Model được network topology
- Node embeddings học behavior patterns
- Anti-leakage IP randomization

---

## 🔍 Troubleshooting

### Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 2048  # giảm từ 4096

# Reduce max_samples
data:
  max_samples: 50000

# Reduce neighbor sampling
training:
  num_neighbors: [10, 5]  # giảm từ [15, 10]
```

### Overfitting
```yaml
# Increase dropout
endpoint_model:
  dropout: 0.3  # tăng từ 0.2

# Add weight decay
training:
  weight_decay: 0.001  # tăng từ 0.0001

# Enable anti-leakage
endpoint_graph:
  anti_leakage:
    enabled: true
```

### Underfitting
```yaml
# Increase model capacity
endpoint_model:
  hidden_dim: 256  # tăng từ 128
  num_layers: 3    # tăng từ 2

# Increase training epochs
training:
  epochs: 100

# Reduce dropout
endpoint_model:
  dropout: 0.1
```

### Class Imbalance
```yaml
# Enable class weights (already default)
training:
  loss:
    use_class_weights: true

# Hoặc oversample minority class trong preprocessing
```

### Slow Training
```yaml
# Increase batch size (nếu có RAM)
training:
  batch_size: 8192

# Reduce neighbor sampling
training:
  num_neighbors: [10, 5]

# Use GPU
project:
  device: "cuda"  # hoặc "mps" cho Mac M1/M2
```

---

## 📚 Technical Details

### Data Preprocessing Pipeline

**Step-by-step:**
1. **Load CSV**: pd.read_csv()
2. **Drop columns**: Remove metadata (Flow ID, Timestamp, IPs, Ports)
3. **Handle inf/nan**: Replace với 0 hoặc median
4. **Encode labels**: LabelEncoder (benign=0, attack=1)
5. **Split data**: 60% train, 10% val, 30% test (stratified)
6. **Scale features**: StandardScaler hoặc MinMaxScaler
7. **Convert to tensors**: torch.tensor()

### Graph Construction Details

**KNN Graph (Flow-based):**
- Compute pairwise distances (cosine/euclidean)
- For each node, connect k nearest neighbors
- Symmetrize: if A→B then B→A
- Edge weights = similarity scores

**Endpoint Graph (E-GraphSAGE):**
- Extract unique endpoints (IP:Port)
- Create node index mapping
- Build edge_index from flow records
- Edge features = flow features
- Node features = learned embeddings (init zeros)

### Model Architecture Details

**FlowGraphSAGE Layers:**
```
Layer 1: SAGEConv(78 → 128)
  - W_self: (78, 128)
  - W_neigh: (78, 128)
  - Operation: concat(W_self·x, W_neigh·mean(neighbors))

Layer 2: SAGEConv(128 → 128)
  - Similar structure

Classifier: Linear(128 → 2)
  - W: (128, 2)
  - b: (2,)

Total params: ~25K
```

**E-GraphSAGE Layers:**
```
Layer 1: EdgeFeatureSAGEConv(78 → 128)
  - Aggregate edge features instead of node features
  
Layer 2: EdgeFeatureSAGEConv(128 → 128)

Edge Classifier:
  Linear(256 → 128) + ReLU + Dropout
  Linear(128 → 2)

Total params: ~60K
```

### Training Algorithm

**Pseudocode:**
```python
for epoch in range(max_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_metrics = evaluate(model, val_loader)
    
    # Early stopping check
    if early_stopping(val_metrics['f1']):
        break
    
    # Save best model
    if val_metrics['f1'] > best_f1:
        save_checkpoint(model, optimizer, epoch)
```

### Loss Functions

**Cross Entropy Loss:**
```python
loss = -∑ y_i · log(p_i)

# With class weights
loss = -∑ w_i · y_i · log(p_i)
where w_i = class_weight[class_i]
```

**Class Weights Computation:**
```python
# Balanced class weights
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(classes) * class_counts)

# Example:
# Class 0 (benign): 80,000 samples → weight = 0.625
# Class 1 (attack): 20,000 samples → weight = 2.5
```

---

## 🎓 References

### Papers
1. **GraphSAGE**: Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
2. **E-GraphSAGE**: Lo et al. "E-GraphSAGE: A Graph Neural Network based Intrusion Detection System" (arXiv 2022)
3. **CICIDS2018**: Sharafaldin et al. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"

### Datasets
- **CICIDS2018**: [UNB Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
- Contains 80+ features extracted from network flows
- Multiple attack types: DoS, DDoS, Brute Force, Web Attacks, etc.

### Key Concepts

**Graph Neural Networks:**
- Message passing: Nodes exchange information with neighbors
- Aggregation: Combine neighbor information
- Update: Update node representations

**Intrusion Detection:**
- Binary classification: Benign vs Attack
- Multiclass: Classify attack types
- Metrics: FAR (False Alarm Rate), DR (Detection Rate)

**Class Imbalance:**
- Network traffic datasets thường imbalanced (nhiều benign hơn attack)
- Solutions: Weighted loss, oversampling, focal loss

**Anti-leakage:**
- IP address có thể become "shortcut" features
- Model học thuộc IP thay vì behavior patterns
- Solution: IP randomization trước khi training

---

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/

# Lint
flake8 src/
```

### Adding New Features

**New GNN Model:**
1. Create model class in `src/gnn_ids/model/`
2. Implement `forward()` method
3. Update `config.yaml` với model parameters
4. Add training loop nếu cần

**New Metric:**
1. Add metric function trong `src/gnn_ids/utils/metrics.py`
2. Update `compute_metrics()` để include metric
3. Update logging và visualization

**New Dataset:**
1. Add loader function trong `src/gnn_ids/data/load_csv.py`
2. Handle dataset-specific preprocessing
3. Update `config.yaml` schema nếu cần

---

## 📝 License

MIT License - see LICENSE file

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- PyTorch Geometric team cho excellent GNN library
- UNB CICIDS2018 dataset creators
- GraphSAGE và E-GraphSAGE paper authors

---

## 📊 Project Statistics

```
Language:     Python
Framework:    PyTorch, PyG
Lines of Code: ~3000
Modules:      15+
Documentation: Comprehensive
```

---

## 🔮 Future Work

**Planned Features:**
- [ ] More GNN architectures (GAT, GCN, GIN)
- [ ] Explainability (GNNExplainer, attention visualization)
- [ ] Real-time inference API
- [ ] Web dashboard cho monitoring
- [ ] Multi-dataset support
- [ ] AutoML cho hyperparameter tuning
- [ ] Distributed training support

**Research Directions:**
- [ ] Temporal graph networks cho time-series analysis
- [ ] Heterogeneous graphs (multiple node/edge types)
- [ ] Few-shot learning cho rare attack types
- [ ] Transfer learning across datasets

---

**Last Updated**: December 2025  
**Version**: 1.0.0

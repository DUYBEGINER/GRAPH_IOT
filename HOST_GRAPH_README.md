# Host-Connection Graph for Network Intrusion Detection

## Overview

This implementation provides a **Host-Connection Graph** (also called IP-Flow Graph) approach for network intrusion detection using Graph Neural Networks (GNNs). Unlike similarity-based graphs, this approach models the actual network topology and communication patterns.

## Graph Structure

### Nodes (Hosts)
- **Representation**: Each unique IP address in the network
- **Features**: Aggregated statistics from all flows involving that host
  - Statistical summaries: mean, std, max, min, sum
  - Flow-level features: packet counts, byte counts, flags, etc.
  
### Edges (Flows)
- **Representation**: Network flows between hosts (directed)
- **Features**: Flow-level characteristics
  - Flow duration, protocol, packet counts
  - Byte transfers, flag counts, etc.

### Labels
- **Node-level labels**: Assigned via majority voting from flows
  - If a host participates mostly in attack traffic → Attack label
  - Otherwise → Benign label

## Key Differences from Similarity-Based Graphs

| Aspect | Similarity-Based Graph | Host-Connection Graph |
|--------|----------------------|----------------------|
| **Nodes** | Individual flows | IP addresses (hosts) |
| **Edges** | k-NN or cosine similarity | Actual network connections |
| **Interpretability** | Abstract similarity | Real network topology |
| **Scalability** | O(n²) or O(nk) | O(m) where m = #flows |
| **Domain Relevance** | Generic ML | Network-specific |

## Files

### 1. `build_host_graph.py`
Constructs the Host-Connection Graph from raw CSV data.

**Key Features:**
- Extracts IP addresses from network flows
- Aggregates flow statistics per host
- Creates directed edges representing flows
- Assigns labels via majority voting

**Configuration:**
```python
DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
CSV_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
MAX_FLOWS = 200000  # Limit flows to process
MIN_FLOW_COUNT = 2  # Minimum flows per host
```

**Usage:**
```bash
python build_host_graph.py
```

**Output:**
- `graph_data/host_graph.pt` - PyTorch Geometric graph
- `graph_data/host_graph_metadata.pkl` - Metadata (IP mappings, labels, etc.)

### 2. `train_host_gnn.py`
Trains GNN models on the Host-Connection Graph.

**Supported Models:**
- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network) - Default
- **GraphSAGE** (Graph Sample and Aggregate)

**Configuration:**
```python
MODEL_TYPE = 'GAT'  # 'GCN', 'GAT', or 'GraphSAGE'
TASK = 'binary'  # 'binary' or 'multi'
EPOCHS = 200
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.5
```

**Usage:**
```bash
python train_host_gnn.py
```

**Output:**
- `models/host_gnn_gat_binary_best.pt` - Best model weights
- `results/host_gnn_gat_binary_history.png` - Training curves
- `results/host_gnn_gat_binary_confusion_matrix.png` - Confusion matrix
- `results/host_gnn_gat_binary_results.pkl` - Detailed results

## Architecture

### Graph Construction Pipeline

```
Raw CSV Data (with IP addresses)
         ↓
Extract Src IP, Dst IP, Features, Labels
         ↓
Create IP → Node ID mapping
         ↓
Aggregate flow statistics per IP
         ↓
Create edges from flows
         ↓
Assign labels via majority voting
         ↓
PyTorch Geometric Data object
```

### GNN Model Architecture

```
Input: Node Features (70 aggregated features)
         ↓
GNN Layer 1 (Input → Hidden)
         ↓
Batch Norm + ReLU + Dropout
         ↓
GNN Layer 2 (Hidden → Hidden)
         ↓
Batch Norm + ReLU + Dropout
         ↓
GNN Layer 3 (Hidden → Output)
         ↓
Log Softmax
         ↓
Output: Node Classifications (Benign/Attack)
```

## Node Feature Aggregation

For each host (IP address), we compute the following statistics from all associated flows:

### Aggregated Features (5 statistics × 14 flow features = 70 features)

**Flow Features:**
1. Flow Duration
2. Total Forward Packets
3. Total Backward Packets
4. Total Forward Bytes
5. Total Backward Bytes
6. Flow Bytes/s
7. Flow Packets/s
8. Packet Length Mean
9. Packet Length Std
10. FIN Flag Count
11. SYN Flag Count
12. RST Flag Count
13. PSH Flag Count
14. ACK Flag Count

**Statistics per Feature:**
1. **Mean**: Average value
2. **Std**: Standard deviation
3. **Max**: Maximum value
4. **Min**: Minimum value
5. **Sum**: Total sum

## Edge Features

Each edge (flow) has 13 features:
1. Flow Duration
2. Protocol
3. Total Forward Packets
4. Total Backward Packets
5. Total Forward Bytes
6. Total Backward Bytes
7. Flow Bytes/s
8. Flow Packets/s
9. FIN Flag Count
10. SYN Flag Count
11. RST Flag Count
12. PSH Flag Count
13. ACK Flag Count

## Example Results

Based on the Thuesday-20-02-2018 dataset:

```
Graph Statistics:
- Nodes (Hosts): 19
- Edges (Flows): 199,983
- Node Features: 70
- Edge Features: 13
- Average Degree: 10,525.42

Label Distribution:
- Benign: 8 hosts (42.1%)
- DDoS attacks-LOIC-HTTP: 11 hosts (57.9%)
```

## Advantages of Host-Connection Graph

### 1. **Network Topology Awareness**
- Models real network structure
- Captures communication patterns
- Preserves client-server relationships

### 2. **Interpretability**
- Nodes are actual IP addresses
- Edges are actual network flows
- Easy to trace back to original traffic

### 3. **Scalability**
- Nodes = O(# unique IPs) << # flows
- Suitable for large-scale networks
- Can process millions of flows

### 4. **Attack Detection**
- **C&C Communication**: Detect hosts communicating with C&C servers
- **Port Scanning**: Identify hosts connecting to many destinations
- **DDoS Sources**: Find hosts with abnormally high out-degree
- **Botnet Members**: Detect hosts with similar communication patterns

### 5. **Graph-based Features**
- **Degree Centrality**: Hosts with many connections
- **Betweenness**: Hosts acting as bridges
- **PageRank**: Important/central hosts
- **Community Detection**: Identify groups of related hosts

## Training Tips

### 1. **Graph Preprocessing**
```python
# Make undirected (optional)
data.edge_index = to_undirected(data.edge_index)

# Add self-loops
data.edge_index, _ = add_self_loops(data.edge_index)

# Normalize features
data.x = (data.x - data.x.mean(0)) / data.x.std(0)
```

### 2. **Model Selection**
- **Small graphs (<1000 nodes)**: GAT (attention mechanism)
- **Medium graphs**: GCN (simple and effective)
- **Large graphs**: GraphSAGE (sampling-based)

### 3. **Hyperparameters**
- **Learning Rate**: 0.001 - 0.01
- **Hidden Dim**: 32 - 128
- **Layers**: 2 - 4
- **Dropout**: 0.3 - 0.6

### 4. **Handling Imbalanced Data**
```python
# Weighted loss
criterion = torch.nn.NLLLoss(weight=torch.tensor(class_weights))
```
- Nodes: 50,000 flows
- Edges: k-NN connections (k=10)
- Features: 70 flow features
- Labels: Flow-level (Benign/Attack)
```

### Host-Connection Graph (New)
```python
# build_host_graph.py
- Nodes: 19 hosts (IPs)
- Edges: 199,983 actual flows
- Features: 70 aggregated features per host
- Labels: Host-level (majority vote)
```

## Use Cases

### 1. **Malicious Host Detection**
Identify compromised or malicious hosts in the network based on their communication patterns.

### 2. **Lateral Movement Detection**
Detect attackers moving between hosts within the network.

### 3. **Botnet Detection**
Identify groups of hosts exhibiting coordinated behavior.

### 4. **Anomaly Detection**
Find hosts with unusual communication patterns compared to their neighbors.

### 5. **Network Segmentation**
Discover natural communities in the network for security zoning.

## Future Enhancements

1. **Temporal Graphs**: Add time dimension to track evolution
2. **Heterogeneous Graphs**: Include different node types (servers, clients, routers)
3. **Edge Prediction**: Predict future connections for proactive defense
4. **Graph Pooling**: Hierarchical graph representation
5. **Multi-Task Learning**: Simultaneously predict multiple attack types

## References
1. **Temporal Graphs**: Add time dimension to track evolution
2. **Heterogeneous Graphs**: Include different node types (servers, clients, routers)
3. **Edge Prediction**: Predict future connections for proactive defense
4. **Graph Pooling**: Hierarchical graph representation
5. **Multi-Task Learning**: Simultaneously predict multiple attack types

The Host-Connection Graph approach provides a more natural and interpretable representation of network traffic for intrusion detection. By modeling actual network topology and communication patterns, GNNs can leverage structural information to improve detection accuracy and provide actionable insights for security analysts.


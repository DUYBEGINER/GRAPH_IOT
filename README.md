# GNN-based Network Anomaly Detection for CICIDS2018

This project implements GraphSAGE models for detecting network anomalies (Benign vs Attack) using the CICIDS2018 dataset.

## Project Structure

```
IOT/
├── GNN_Flow/                    # Flow-based approach
│   ├── preprocess.py            # Data preprocessing
│   ├── build_graph.py           # Graph construction using FAISS KNN
│   ├── train.py                 # Model training
│   ├── inference.py             # Local inference/demo
│   ├── kaggle_notebook.py       # Complete pipeline for Kaggle
│   └── requirements.txt
│
├── GNN_IP/                      # IP-based approach
│   ├── preprocess.py            # Data preprocessing
│   ├── build_graph.py           # Graph construction using IP connections
│   ├── train.py                 # Model training
│   ├── inference.py             # Local inference/demo
│   ├── kaggle_notebook.py       # Complete pipeline for Kaggle
│   └── requirements.txt
│
└── CICIDS2018-CSV/              # Dataset directory
    └── *.csv
```

## Two Approaches

### 1. Flow-based Approach (GNN_Flow)
- **Data**: All files EXCEPT `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`
- **Graph Construction**: Each flow is a node, connected via K-Nearest Neighbors using FAISS
- **Use Case**: When IP information is not available

### 2. IP-based Approach (GNN_IP)
- **Data**: Only `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` (contains IP info)
- **Graph Construction**: Each IP is a node, edges are communication flows between IPs
- **Node Features**: Aggregated flow statistics (mean, std, max)
- **Use Case**: When you want to identify malicious IPs

## Running on Kaggle

### Flow-based:
1. Upload CICIDS2018-CSV dataset to Kaggle
2. Create new notebook with GPU enabled
3. Copy content of `GNN_Flow/kaggle_notebook.py`
4. Run all cells
5. Download `gnn_flow_output.zip`

### IP-based:
1. Upload CICIDS2018-CSV dataset to Kaggle
2. Create new notebook with GPU enabled
3. Copy content of `GNN_IP/kaggle_notebook.py`
4. Run all cells
5. Download `gnn_ip_output.zip`

## Local Inference

### Flow-based:
```bash
python GNN_Flow/inference.py --model-dir path/to/output --csv path/to/test.csv --output predictions.csv
```

### IP-based:
```bash
python GNN_IP/inference.py --model-dir path/to/output --csv path/to/test.csv --output predictions.json
```

## Model Architecture

- **GraphSAGE** with 3 layers
- Hidden channels: 128
- Dropout: 0.3
- Binary classification (Benign vs Attack)

## Output Files

After training, you get:
- `models/best_model.pt` - Trained model weights
- `scaler.pkl` - Feature scaler
- `feature_names.pkl` - Feature names list
- `results/results.json` - Evaluation metrics
- `results/training_history.png` - Training curves
- `results/confusion_matrix.png` - Confusion matrix visualization
- `results/metrics.png` - Performance metrics chart

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- FAISS (for Flow-based approach)
- scikit-learn
- pandas
- numpy
- matplotlib


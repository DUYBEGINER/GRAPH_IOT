# Demo Script - IP GNN Realtime Detection

## Mô tả

Script `demo_realtime_detect.py` là ứng dụng Streamlit để demo realtime detection sử dụng model IP-GNN (E-GraphSAGE) được train từ notebook `kaggle_ip_gnn.ipynb`.

### Kiến trúc

- **Model**: E-GraphSAGE (Edge-based GraphSAGE)
- **Task**: Edge classification (phân loại flows)
- **Node**: Endpoint (IP addresses)
- **Edge**: Network flows
- **Features**: Edge features (flow statistics)

## Yêu cầu

### Dependencies
```bash
pip install torch torch-geometric streamlit pandas numpy scikit-learn
```

### Model & Data
Cần có:
- Model đã train: `ip_gnn/ip_gnn_result/best_model.pt`
- Feature names: `dataset-processed/ip_gnn/feature_names.json`
- Scaler: `dataset-processed/ip_gnn/scaler.pkl`

## Cách sử dụng

### 1. Chạy Streamlit App

```bash
cd /Users/tphuc263/Project/clone/GRAPH_IOT
streamlit run demo_scripts/demo_realtime_detect.py
```

### 2. Cấu hình trong Sidebar

**Model Parameters:**
- `Hidden dim`: Số chiều hidden layer (mặc định: 128)
- `Num layers`: Số layer GNN (mặc định: 2)
- `Dropout`: Tỷ lệ dropout (mặc định: 0.2)
- `Classification threshold`: Ngưỡng phân loại (mặc định: 0.5)

**Inference Settings:**
- `Label column`: Tên cột label trong CSV (mặc định: "Label")
- `Chunk size`: Số dòng xử lý mỗi lần (mặc định: 2000)
- `Window size`: Kích thước cửa sổ buffer (mặc định: 8000)
- `Speed`: Độ trễ giữa các chunk (mặc định: 0.2s)

### 3. Upload CSV File

Upload file CSV với định dạng tương tự dataset training:
- Ví dụ: `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv`
- File phải có các cột feature tương ứng với model
- (Optional) Cột `Label` để tính accuracy

### 4. Xem kết quả

App sẽ hiển thị:
- **Metrics**: Accuracy, Attack Rate, Rows Processed
- **Latest Predictions**: Bảng dự đoán mới nhất
- **Performance Chart**: Biểu đồ theo thời gian
- **Attack Logs**: 
  - True Positive (Detected Attacks)
  - False Positive (False Alarms)
  - False Negative (Missed Attacks)

### 5. Download Results

Sau khi xử lý xong, có thể download:
- `attack_detected_true_positive.csv`
- `attack_false_positive.csv`
- `attack_missed_false_negative.csv`

## Khác biệt với Flow-GNN

| Aspect | Flow-GNN (cũ) | IP-GNN (mới) |
|--------|---------------|--------------|
| Model | FlowGraphSAGE | E-GraphSAGE |
| Task | Node classification | Edge classification |
| Node | Flow | Endpoint (IP) |
| Edge | KNN graph | Network flow |
| Features | Node features | Edge features |
| Graph construction | KNN | Direct from flows |

## Lưu ý

- Script tự động sử dụng CPU để tránh MPS segfault trên macOS
- Model phải được train với cùng kiến trúc (EGraphSAGE)
- Feature dimension phải khớp với model checkpoint
- Graph construction khác với Flow-GNN (không dùng KNN)

## Troubleshooting

### Lỗi: "Cannot load model checkpoint"
- Kiểm tra path `MODEL_DIR` và `DATA_DIR`
- Đảm bảo file `best_model.pt` tồn tại

### Lỗi: "Feature dimension mismatch"
- Kiểm tra `feature_names.json` khớp với model
- Đảm bảo CSV có đủ các cột feature

### Lỗi: "Out of memory"
- Giảm `chunk_size` hoặc `window_size`
- Đảm bảo đang dùng CPU (không dùng GPU)

## Tác giả

Script được cập nhật để tương thích với model IP-GNN từ notebook `kaggle_ip_gnn.ipynb`.

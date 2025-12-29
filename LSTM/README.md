# 🧠 LSTM-based IoT Anomaly Detection

Module này sử dụng mạng **Long Short-Term Memory (LSTM)** để phát hiện tấn công trong mạng IoT dựa trên dataset CICIDS2018.

## 📂 Cấu trúc thư mục

```
LSTM/
├── preprocess_lstm.py    # Clean dữ liệu & Tạo chuỗi (Sliding Window)
├── train_lstm.py         # Huấn luyện mô hình & Đánh giá (Confusion Matrix, ROC)
├── inference_lstm.py     # Dự đoán trên file CSV mới & Visualize kết quả
├── kaggle_notebook.ipynb # Notebook tích hợp sẵn để chạy trên Kaggle GPU
├── README.md             # Hướng dẫn sử dụng
├── processed_lstm/       # Lưu trữ dữ liệu sau tiền xử lý (.npy, scaler.pkl)
└── models/               # Lưu trữ model (.keras) và biểu đồ huấn luyện
```

## 🚀 Hướng dẫn nhanh

### 1. Chuẩn bị dữ liệu
Đặt các file CSV của dataset CICIDS2018 vào thư mục `GRAPH_IOT/data_IOT/`.

### 2. Tiền xử lý dữ liệu (Local)
LSTM cần dữ liệu đầu vào dạng 3D `(Samples, TimeSteps, Features)`. Chạy lệnh sau để clean dữ liệu thô và tạo sliding window:
```bash
python preprocess_lstm.py
```
*   **Kết quả:** Tạo ra các file `.npy` trong `processed_lstm/` và bộ chuẩn hóa `scaler.pkl`.
*   **Cấu hình:** Mặc định `WINDOW_SIZE = 10`.

### 3. Huấn luyện mô hình
```bash
python train_lstm.py
```
*   Tự động sử dụng GPU nếu có.
*   Hỗ trợ **Mixed Precision** để tăng tốc trên các card đồ họa đời mới (T4, RTX...).

### 4. Dự đoán và Trực quan hóa
```bash
python inference_lstm.py
```
*   Script sẽ chọn một file có chứa dấu hiệu tấn công (như DDoS) để demo khả năng phát hiện của model qua biểu đồ.

## ⚙️ Cấu hình quan trọng

Các tham số có thể tùy chỉnh trong code:
*   `WINDOW_SIZE`: Số lượng flow nhìn lại quá khứ (mặc định: 10).
*   `TARGET_ROWS`: Số lượng dòng dữ liệu thô tối đa được load (mặc định: 3,500,000).
*   `BALANCE_DATA`: Tự động cân bằng giữa mẫu tấn công và mẫu bình thường (Undersampling).

## 📊 Kết quả đạt được
Model sẽ xuất ra các biểu đồ trong thư mục `models/`:
*   `training_history.png`: Biểu đồ Accuracy/Loss qua các Epoch.
*   `evaluation_metrics.png`: Confusion Matrix, ROC Curve và Precision-Recall Curve.
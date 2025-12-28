# 🧠 LSTM-based IoT Anomaly Detection

Module này sử dụng mạng **Long Short-Term Memory (LSTM)** để phát hiện tấn công trong mạng IoT dựa trên dataset CICIDS2018.

## 📂 Cấu trúc thư mục

```
LSTM/
├── preprocess_lstm.py    # Chuẩn bị dữ liệu dạng chuỗi (Sliding Window)
├── train_lstm.py         # Huấn luyện mô hình LSTM
├── inference_lstm.py     # (To do) Dự đoán dữ liệu mới
├── README.md             # Hướng dẫn sử dụng
└── models/               # Nơi lưu model đã train
```

## 🚀 Hướng dẫn nhanh

### 1. Chuẩn bị dữ liệu gốc
Đảm bảo bạn đã chạy script xử lý dữ liệu chung ở thư mục gốc trước:
```bash
cd ..
python preprocess_data.py
cd LSTM
```
*Điều này tạo ra folder `../processed_data/` chứa dữ liệu sạch.*

### 2. Tạo dữ liệu chuỗi (Sequence Data)
LSTM cần dữ liệu đầu vào dạng 3D `(Samples, TimeSteps, Features)`. Chạy lệnh sau để tạo cửa sổ trượt (Sliding Window):

```bash
python preprocess_lstm.py
```
*Cấu hình mặc định: Window Size = 10 (nhìn lại 10 flows trước đó).*

### 3. Huấn luyện mô hình
```bash
python train_lstm.py
```

## ⚙️ Cấu hình (Configuration)

Bạn có thể chỉnh sửa các tham số trong `preprocess_lstm.py` và `train_lstm.py`:

*   **WINDOW_SIZE**: Số lượng time steps (mặc định: 10).
*   **BATCH_SIZE**: Kích thước batch (mặc định: 64).
*   **EPOCHS**: Số vòng lặp huấn luyện (mặc định: 20).
*   **LSTM_UNITS**: Số lượng noron trong lớp LSTM.

## 📊 Kiến trúc Model

*   **Input Layer**: Shape `(Window_Size, Features)`
*   **LSTM Layer 1**: Trích xuất đặc trưng chuỗi.
*   **Dropout**: Chống Overfitting.
*   **LSTM Layer 2**: (Optional) Học các patterns phức tạp hơn.
*   **Dense Layer**: Phân lớp (Sigmoid cho Binary, Softmax cho Multi-class).

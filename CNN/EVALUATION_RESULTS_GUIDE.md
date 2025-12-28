# 📊 HƯỚNG DẪN KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CNN

## Tổng quan
Sau khi train mô hình CNN xong, hệ thống sẽ tự động lưu đầy đủ các kết quả đánh giá vào thư mục `models/`.

---

## 📁 CÁC FILE KẾT QUẢ ĐƯỢC LƯU

### 1. **evaluation_results.json** ⭐
File JSON chứa tất cả các metrics đánh giá mô hình:

#### Metrics trên Test Set:
- `test_loss`: Loss trên test set
- `test_accuracy`: Độ chính xác tổng thể
- `test_precision`: Precision tổng thể
- `test_recall`: Recall tổng thể
- `test_f1_score`: F1-Score tổng thể

#### Metrics chi tiết cho từng class:
- `benign_precision`, `benign_recall`, `benign_f1`: Metrics cho class Benign
- `attack_precision`, `attack_recall`, `attack_f1`: Metrics cho class Attack

#### Confusion Matrix:
- `confusion_matrix`: Ma trận nhầm lẫn dạng 2D array
- `true_negative` (TN): Số mẫu Benign dự đoán đúng
- `false_positive` (FP): Số mẫu Benign dự đoán nhầm thành Attack
- `false_negative` (FN): Số mẫu Attack dự đoán nhầm thành Benign
- `true_positive` (TP): Số mẫu Attack dự đoán đúng

#### Metrics bổ sung:
- `specificity`: Tỷ lệ nhận diện đúng Benign (TN / (TN + FP))
- `false_positive_rate`: Tỷ lệ dự đoán nhầm Benign thành Attack
- `false_negative_rate`: Tỷ lệ bỏ sót Attack

#### Thông tin về Training:
- `training_time_seconds`: Thời gian train (giây)
- `training_time_minutes`: Thời gian train (phút)
- `epochs_trained`: Số epochs đã train
- `best_val_loss`: Val loss tốt nhất
- `best_val_accuracy`: Val accuracy cao nhất
- `best_val_precision`: Val precision cao nhất
- `best_val_recall`: Val recall cao nhất
- `best_val_f1_score`: Val F1-Score cao nhất
- `best_val_loss_epoch`: Epoch đạt val_loss tốt nhất
- `best_val_accuracy_epoch`: Epoch đạt val_accuracy cao nhất

#### Classification Report:
- `classification_report`: Chi tiết precision, recall, f1-score cho từng class

---

### 2. **training_history.json** 📈
Lịch sử training qua từng epoch:
- `loss`: Training loss qua các epochs
- `accuracy`: Training accuracy qua các epochs
- `precision`: Training precision qua các epochs
- `recall`: Training recall qua các epochs
- `val_loss`: Validation loss qua các epochs
- `val_accuracy`: Validation accuracy qua các epochs
- `val_precision`: Validation precision qua các epochs
- `val_recall`: Validation recall qua các epochs

**Công dụng**: Dùng để phân tích xu hướng training, phát hiện overfitting/underfitting.

---

### 3. **training_config.json** ⚙️
Cấu hình hyperparameters đã sử dụng khi train:
- `batch_size`: Kích thước batch
- `epochs`: Số epochs tối đa
- `learning_rate`: Learning rate ban đầu
- `dropout_rate`: Tỷ lệ dropout
- `patience`: Early stopping patience
- `random_seed`: Random seed để reproducibility
- `tensorflow_version`: Phiên bản TensorFlow
- `created_at`: Thời gian tạo model

**Công dụng**: Để tái tạo lại kết quả hoặc so sánh các lần training khác nhau.

---

### 4. **y_pred.npy** 🎯
Dự đoán của mô hình trên test set (binary: 0 hoặc 1).

**Cách load**:
```python
import numpy as np
y_pred = np.load('models/y_pred.npy')
print(y_pred.shape)  # (n_test_samples,)
print(y_pred[:10])   # [0, 1, 0, 0, 1, ...]
```

---

### 5. **y_pred_prob.npy** 📊
Xác suất dự đoán (probability) của mô hình trên test set.

**Cách load**:
```python
import numpy as np
y_pred_prob = np.load('models/y_pred_prob.npy')
print(y_pred_prob.shape)  # (n_test_samples, 1)
print(y_pred_prob[:10])   # [[0.0234], [0.9876], [0.1234], ...]
```

**Công dụng**: 
- Điều chỉnh threshold (thay vì 0.5)
- Vẽ ROC curve, PR curve
- Phân tích confidence của predictions

---

### 6. **training_history.png** 📉
Biểu đồ 4 subplot:
1. **Loss**: Train loss vs Val loss
2. **Accuracy**: Train accuracy vs Val accuracy
3. **Precision**: Train precision vs Val precision
4. **Recall**: Train recall vs Val recall

**Công dụng**: Trực quan hóa quá trình training, phát hiện overfitting.

---

### 7. **confusion_matrix.png** 🔲
Confusion matrix dạng heatmap với số lượng thực tế:
```
                 Predicted
                 Benign  Attack
Actual Benign     TN      FP
Actual Attack     FN      TP
```

---

### 8. **confusion_matrix_normalized.png** 📊
Confusion matrix được normalize theo tỷ lệ phần trăm của từng class.

**Công dụng**: Dễ nhìn hơn khi các class không cân bằng.

---

### 9. **best_model.keras** 💾
Model tốt nhất (val_loss thấp nhất) được lưu tự động bởi ModelCheckpoint.

**Cách load**:
```python
from tensorflow import keras
model = keras.models.load_model('models/best_model.keras')
```

---

### 10. **final_model.keras** 🏁
Model cuối cùng sau khi kết thúc training.

---

### 11. **model_weights.h5** ⚖️
Chỉ chứa weights của model (không có architecture).

**Cách load**:
```python
model.load_weights('models/model_weights.h5')
```

---

## 📖 CÁCH ĐỌC VÀ PHÂN TÍCH KẾT QUẢ

### 1. Đọc evaluation_results.json
```python
import json

with open('models/evaluation_results.json', 'r') as f:
    results = json.load(f)

print(f"Test Accuracy: {results['test_accuracy']:.4f}")
print(f"Test Precision: {results['test_precision']:.4f}")
print(f"Test Recall: {results['test_recall']:.4f}")
print(f"Test F1-Score: {results['test_f1_score']:.4f}")

print(f"\nBenign - Precision: {results['benign_precision']:.4f}, Recall: {results['benign_recall']:.4f}")
print(f"Attack - Precision: {results['attack_precision']:.4f}, Recall: {results['attack_recall']:.4f}")

print(f"\nConfusion Matrix:")
cm = results['confusion_matrix']
print(f"TN={cm[0][0]}, FP={cm[0][1]}")
print(f"FN={cm[1][0]}, TP={cm[1][1]}")
```

### 2. Đọc training_history.json
```python
import json
import matplotlib.pyplot as plt

with open('models/training_history.json', 'r') as f:
    history = json.load(f)

# Vẽ loss
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### 3. Phân tích predictions
```python
import numpy as np

y_test = np.load('training_data/y_test.npy')
y_pred = np.load('models/y_pred.npy')
y_pred_prob = np.load('models/y_pred_prob.npy')

# Tìm những mẫu dự đoán sai
wrong_indices = np.where(y_test != y_pred)[0]
print(f"Số mẫu dự đoán sai: {len(wrong_indices)}")

# Xem confidence của những mẫu dự đoán sai
wrong_probs = y_pred_prob[wrong_indices]
print(f"Confidence trung bình của dự đoán sai: {wrong_probs.mean():.4f}")
```

---

## 🎯 METRICS QUAN TRỌNG CẦN CHÚ Ý

### 1. **Accuracy** (Độ chính xác tổng thể)
- Tốt khi dataset cân bằng
- **Không đáng tin cậy** khi dataset mất cân bằng (ví dụ: 95% Benign, 5% Attack)

### 2. **Precision** (Độ chính xác của dự đoán dương)
- **Attack Precision**: Trong số các mẫu dự đoán là Attack, bao nhiêu % thực sự là Attack?
- **Quan trọng** khi muốn giảm False Positive (cảnh báo nhầm)

### 3. **Recall (Sensitivity)** (Tỷ lệ phát hiện được)
- **Attack Recall**: Trong số các mẫu thực sự là Attack, bao nhiêu % được phát hiện?
- **Cực kỳ quan trọng** trong bài toán phát hiện tấn công (muốn giảm False Negative - bỏ sót tấn công)

### 4. **F1-Score** (Trung bình điều hòa của Precision và Recall)
- Cân bằng giữa Precision và Recall
- Tốt khi dataset mất cân bằng

### 5. **Specificity (True Negative Rate)**
- Khả năng nhận diện đúng lưu lượng Benign
- Quan trọng để tránh cảnh báo nhầm quá nhiều

---

## 🚨 DẤU HIỆU CẦN CHÚ Ý

### ⚠️ Overfitting:
- Train accuracy cao, Val accuracy thấp hơn nhiều
- Train loss giảm liên tục, Val loss tăng

**→ Giải pháp**: Tăng Dropout, thêm Regularization, giảm số epochs

### ⚠️ Underfitting:
- Cả Train và Val accuracy đều thấp
- Loss không giảm nữa

**→ Giải pháp**: Tăng model complexity, train lâu hơn, tăng learning rate

### ⚠️ Class Imbalance:
- Accuracy cao nhưng Attack Recall thấp
- Model chỉ dự đoán Benign

**→ Giải pháp**: Sử dụng class_weight, oversample Attack class, undersample Benign

---

## 📊 SO SÁNH CÁC LẦN TRAINING

Tạo bảng so sánh các lần training khác nhau:

```python
import json
import pandas as pd

# Load kết quả từ nhiều lần training
results_1 = json.load(open('models_v1/evaluation_results.json'))
results_2 = json.load(open('models_v2/evaluation_results.json'))
results_3 = json.load(open('models_v3/evaluation_results.json'))

df = pd.DataFrame({
    'Model': ['v1', 'v2', 'v3'],
    'Test Accuracy': [results_1['test_accuracy'], results_2['test_accuracy'], results_3['test_accuracy']],
    'Test F1': [results_1['test_f1_score'], results_2['test_f1_score'], results_3['test_f1_score']],
    'Attack Recall': [results_1['attack_recall'], results_2['attack_recall'], results_3['attack_recall']],
    'Training Time (min)': [results_1['training_time_minutes'], results_2['training_time_minutes'], results_3['training_time_minutes']]
})

print(df)
```

---

## 🎓 KẾT LUẬN

**File `evaluation_results.json`** là file quan trọng nhất, chứa tất cả metrics cần thiết để đánh giá mô hình.

**Với bài toán phát hiện tấn công**, cần ưu tiên:
1. **Attack Recall** (cao) - Phát hiện được nhiều tấn công
2. **Attack Precision** (cao) - Ít cảnh báo nhầm
3. **F1-Score** (cân bằng)

Nếu phải chọn, **Attack Recall** quan trọng hơn vì bỏ sót tấn công nguy hiểm hơn cảnh báo nhầm!

---

**📌 Lưu ý**: Tất cả các file kết quả được lưu tự động sau khi chạy `step3_train_cnn.py`. Không cần làm gì thêm!


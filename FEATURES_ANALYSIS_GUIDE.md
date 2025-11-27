# 📋 HƯỚNG DẪN SỬ DỤNG SCRIPT LIỆT KÊ FEATURES

## 📄 Tổng quan

Script **list_features_pandas.py** giúp bạn phân tích và liệt kê tất cả các đặc trưng (features/columns) trong dataset CICIDS2018 một cách chi tiết.

## ✨ Tính năng

### 🔍 Phân tích chi tiết mỗi file:
- ✅ Số lượng dòng (rows)
- ✅ Số lượng cột (columns)
- ✅ Tên tất cả các cột
- ✅ Kiểu dữ liệu của từng cột (int, float, object)
- ✅ Số lượng missing values trong từng cột
- ✅ Dung lượng RAM sử dụng (MB)

### 📊 Thống kê tổng quan:
- ✅ Tổng số file xử lý
- ✅ Tổng số features duy nhất
- ✅ Tổng số dòng dữ liệu
- ✅ Tổng dung lượng RAM
- ✅ Kiểm tra schema consistency (các file có cùng cột không)
- ✅ Phân bố kiểu dữ liệu (Integer, Float, Object)

### 📝 Output:
- ✅ In kết quả ra màn hình console
- ✅ Lưu chi tiết đầy đủ vào file **features_summary.txt**

## 🚀 Cách sử dụng

### Bước 1: Cài đặt pandas (nếu chưa có)
```bash
pip install pandas
```

### Bước 2: Chạy script
```bash
python list_features_pandas.py
```

### Bước 3: Xem kết quả
- Kết quả tóm tắt hiển thị trên màn hình
- Kết quả chi tiết trong file: `CICIDS2018-CSV/features_summary.txt`

## 🎯 Output mẫu trên màn hình

```
====================================================================================================
SCRIPT LIỆT KÊ CÁC ĐẶC TRƯNG TRONG DATASET CICIDS2018 (PANDAS VERSION)
====================================================================================================

📁 Thư mục dữ liệu: D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV
📊 Tìm thấy 10 file CSV

[1/10] Đang xử lý: Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
    ✓ Số dòng: 1,048,575
    ✓ Số cột: 80
    ✓ RAM: 645.32 MB

[2/10] Đang xử lý: Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
    ✓ Số dòng: 663,809
    ✓ Số cột: 80
    ✓ RAM: 408.15 MB
    ⚠️  3 cột có missing values

... (tiếp tục với các file khác)

====================================================================================================
TÓM TẮT TỔNG QUAN
====================================================================================================
Tổng số file:              10
Tổng số features duy nhất: 80
Tổng số dòng dữ liệu:      6,345,234
Tổng dung lượng RAM:       3,892.45 MB

✓ Tất cả các file có cùng schema (các cột giống nhau và cùng thứ tự)

📋 DANH SÁCH 20 FEATURES ĐẦU TIÊN:
----------------------------------------------------------------------------------------------------
   1. Ack Flag Cnt                                        [int64]
   2. Active Max                                          [float64]
   3. Active Mean                                         [float64]
   4. Active Min                                          [float64]
   5. Active Std                                          [float64]
   6. Bwd Avg Bulk Rate                                   [float64]
   7. Bwd Avg Bytes/Bulk                                  [float64]
   8. Bwd Avg Packets/Bulk                                [float64]
   9. Bwd Header Len                                      [int64]
  10. Bwd IAT Max                                         [float64]
  ... (và 60 features khác)

📊 PHÂN BỐ KIỂU DỮ LIỆU:
----------------------------------------------------------------------------------------------------
  Float                          65 features
  Integer                        10 features
  Object/String                   5 features
```

## 📁 File output: features_summary.txt

File này chứa thông tin chi tiết đầy đủ:

### Phần 1: Chi tiết từng file
```
====================================================================================================
File: Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
====================================================================================================
Số dòng:        1,048,575
Số cột:         80
Dung lượng RAM: 645.32 MB

----------------------------------------------------------------------------------------------------
STT   TÊN CỘT                                          KIỂU DỮ LIỆU         MISSING VALUES    
----------------------------------------------------------------------------------------------------
  1   Dst Port                                         int64                0              
  2   Protocol                                         int64                0              
  3   Flow Duration                                    int64                0              
  4   Tot Fwd Pkts                                     int64                0              
  ... (tiếp tục tất cả các cột)
```

### Phần 2: Tóm tắt tổng quan
- Thống kê tổng số file, features, dòng, RAM
- Kiểm tra schema consistency
- Chi tiết sự khác biệt (nếu có)

### Phần 3: Danh sách tất cả features duy nhất
```
====================================================================================================
DANH SÁCH TẤT CẢ CÁC FEATURES DUY NHẤT
====================================================================================================

STT   TÊN FEATURE                                                  KIỂU DỮ LIỆU PHỔ BIẾN        
----------------------------------------------------------------------------------------------------
  1   Ack Flag Cnt                                                 int64                         
  2   Active Max                                                   float64                       
  3   Active Mean                                                  float64                       
  ... (tất cả features theo thứ tự alphabet)
```

### Phần 4: Phân tích kiểu dữ liệu
```
====================================================================================================
PHÂN TÍCH KIỂU DỮ LIỆU CỦA CÁC FEATURES
====================================================================================================

Phân bố kiểu dữ liệu:
----------------------------------------------------------------------------------------------------
  Float                               65 features
  Integer                             10 features
  Object/String                        5 features
```

## ⚙️ Tùy chỉnh

Mở file `list_features_pandas.py` và chỉnh sửa:

```python
# Thay đổi đường dẫn thư mục data
DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"

# Thay đổi tên file output
OUTPUT_FILE = "features_summary.txt"
```

## 🔧 So sánh với phiên bản simple

| Tính năng | list_features_simple.py | list_features_pandas.py |
|-----------|-------------------------|-------------------------|
| Thư viện | Chỉ dùng CSV (built-in) | Dùng pandas |
| Tốc độ | Nhanh hơn | Chậm hơn một chút |
| Thông tin | Chỉ tên cột | Chi tiết đầy đủ |
| Kiểu dữ liệu | Không | ✅ Có |
| Missing values | Không | ✅ Có |
| Dung lượng RAM | Không | ✅ Có |
| Số dòng data | Không | ✅ Có |

**Khuyến nghị:** Dùng **list_features_pandas.py** để phân tích chi tiết và đầy đủ!

## 📊 Ứng dụng

Script này hữu ích để:
- ✅ Hiểu cấu trúc dataset trước khi train model
- ✅ Kiểm tra tính nhất quán giữa các file
- ✅ Phát hiện missing values
- ✅ Xác định kiểu dữ liệu của features
- ✅ Ước tính dung lượng RAM cần thiết
- ✅ Lập kế hoạch xử lý dữ liệu

## ⚠️ Lưu ý

- Script sẽ đọc toàn bộ mỗi file vào RAM để phân tích
- Với dataset lớn (>1GB mỗi file), cần ít nhất 8GB RAM
- Thời gian xử lý phụ thuộc vào kích thước file (khoảng 1-3 phút/file)
- Nếu gặp lỗi memory, hãy xử lý từng file một hoặc dùng phiên bản simple

## 🎉 Kết quả mong đợi

Sau khi chạy script, bạn sẽ có:
- ✅ Hiểu rõ dataset có 80 features (79 numeric + 1 label)
- ✅ Biết được dataset có khoảng 6-16 triệu dòng
- ✅ Xác định được các cột nào có missing values
- ✅ File text đầy đủ để tham khảo về sau

---

💡 **Tip:** Chạy script này TRƯỚC KHI train model để hiểu rõ dữ liệu!


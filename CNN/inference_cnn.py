"""
======================================================================================
INFERENCE - SỬ DỤNG MÔ HÌNH CNN ĐÃ HUẤN LUYỆN ĐỂ DỰ ĐOÁN
======================================================================================

Script này sử dụng mô hình CNN đã huấn luyện để dự đoán lưu lượng mạng:
- Benign (0): Lưu lượng mạng bình thường
- Attack (1): Lưu lượng mạng bất thường/tấn công

Có thể dự đoán từ file CSV hoặc dữ liệu realtime.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ Lỗi: TensorFlow chưa được cài đặt!")
    sys.exit(1)

# ============================================================================
# KIỂM TRA MÔI TRƯỜNG
# ============================================================================

IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    MODEL_DIR = "/kaggle/working/cnn_results"
    SCALER_PATH = "/kaggle/working/processed_data_cnn/scaler.pkl"
    FEATURE_NAMES_PATH = "/kaggle/working/processed_data_cnn/feature_names.txt"
else:
    MODEL_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\results"
    SCALER_PATH = r"D:\PROJECT\Machine Learning\IOT\CNN\processed_data_cnn\scaler.pkl"
    FEATURE_NAMES_PATH = r"D:\PROJECT\Machine Learning\IOT\CNN\processed_data_cnn\feature_names.txt"

# ============================================================================
# DANH SÁCH CÁC CỘT CẦN LOẠI BỎ (GIỐNG VỚI PREPROCESSING)
# ============================================================================

COLUMNS_TO_DROP = [
    'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Timestamp',
    'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'Label'
]

# ============================================================================
# CLASS INFERENCE
# ============================================================================

class CNNInference:
    """
    Class để thực hiện inference với mô hình CNN đã train
    """

    def __init__(self, model_path=None, scaler_path=SCALER_PATH,
                 feature_names_path=FEATURE_NAMES_PATH):
        """
        Khởi tạo inference engine

        Args:
            model_path: Đường dẫn đến file model (.keras hoặc .h5)
            scaler_path: Đường dẫn đến scaler.pkl
            feature_names_path: Đường dẫn đến feature_names.txt
        """
        # Load model
        if model_path is None:
            model_path = Path(MODEL_DIR) / 'best_model.keras'

        print(f"📂 Đang load model từ: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("✅ Load model thành công!")

        # Load scaler
        print(f"📂 Đang load scaler từ: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✅ Load scaler thành công!")

        # Load feature names
        print(f"📂 Đang load feature names từ: {feature_names_path}")
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ Load {len(self.feature_names)} features thành công!")

    def preprocess_data(self, df):
        """
        Tiền xử lý dữ liệu đầu vào

        Args:
            df: DataFrame chứa dữ liệu cần dự đoán

        Returns:
            X: Numpy array đã được chuẩn hóa và reshape cho CNN
        """
        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip()

        # Loại bỏ cột không cần thiết
        columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Chuyển đổi sang numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Xử lý NaN và Inf
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Đảm bảo có đầy đủ các features theo đúng thứ tự
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            print(f"⚠️ Thiếu features: {missing_features}")
            for feat in missing_features:
                df[feat] = 0

        # Sắp xếp cột theo thứ tự đúng
        df = df[self.feature_names]

        # Chuẩn hóa
        X = self.scaler.transform(df.values)

        # Reshape cho CNN
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return X

    def predict(self, X):
        """
        Dự đoán từ dữ liệu đã tiền xử lý

        Args:
            X: Numpy array đã được tiền xử lý

        Returns:
            predictions: Array các nhãn dự đoán (0=Benign, 1=Attack)
            probabilities: Array các xác suất
        """
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()

        return predictions, probabilities.flatten()

    def predict_from_dataframe(self, df):
        """
        Dự đoán từ DataFrame

        Args:
            df: DataFrame chứa dữ liệu network flow

        Returns:
            DataFrame với cột prediction và probability
        """
        # Tiền xử lý
        X = self.preprocess_data(df.copy())

        # Dự đoán
        predictions, probabilities = self.predict(X)

        # Tạo kết quả
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'label': ['Benign' if p == 0 else 'Attack' for p in predictions]
        })

        return results

    def predict_from_csv(self, csv_path, output_path=None):
        """
        Dự đoán từ file CSV

        Args:
            csv_path: Đường dẫn file CSV đầu vào
            output_path: Đường dẫn file CSV kết quả (optional)

        Returns:
            DataFrame với kết quả dự đoán
        """
        print(f"\n📂 Đang đọc file: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"   Số mẫu: {len(df):,}")

        # Dự đoán
        results = self.predict_from_dataframe(df)

        # Thống kê
        benign_count = (results['prediction'] == 0).sum()
        attack_count = (results['prediction'] == 1).sum()

        print(f"\n📊 KẾT QUẢ DỰ ĐOÁN:")
        print(f"   Benign: {benign_count:,} ({benign_count/len(results)*100:.2f}%)")
        print(f"   Attack: {attack_count:,} ({attack_count/len(results)*100:.2f}%)")

        # Lưu kết quả nếu có output_path
        if output_path:
            results.to_csv(output_path, index=False)
            print(f"\n💾 Đã lưu kết quả: {output_path}")

        return results

    def predict_single(self, features):
        """
        Dự đoán cho một mẫu duy nhất

        Args:
            features: Dictionary hoặc list các giá trị features

        Returns:
            prediction: 0 (Benign) hoặc 1 (Attack)
            probability: Xác suất Attack
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = pd.DataFrame([features], columns=self.feature_names)

        X = self.preprocess_data(df)
        prediction, probability = self.predict(X)

        label = 'Benign' if prediction[0] == 0 else 'Attack'

        return prediction[0], probability[0], label


# ============================================================================
# DEMO USAGE
# ============================================================================

def demo():
    """Demo cách sử dụng inference"""

    print("\n" + "="*80)
    print("🔮 DEMO INFERENCE CNN - PHÁT HIỆN LƯU LƯỢNG MẠNG BẤT THƯỜNG")
    print("="*80)

    # Khởi tạo inference engine
    inference = CNNInference()

    # Demo với test data nếu có
    test_data_path = Path(MODEL_DIR).parent / 'processed_data_cnn' / 'X_test.npy'

    if test_data_path.exists():
        print("\n📊 Demo với test data...")

        # Load test data
        X_test = np.load(test_data_path)
        y_test = np.load(Path(MODEL_DIR).parent / 'processed_data_cnn' / 'y_test.npy')

        # Dự đoán 10 mẫu đầu tiên
        predictions, probabilities = inference.predict(X_test[:10])

        print("\nKết quả dự đoán 10 mẫu đầu:")
        print("-" * 50)
        for i in range(10):
            actual = 'Benign' if y_test[i] == 0 else 'Attack'
            pred = 'Benign' if predictions[i] == 0 else 'Attack'
            correct = '✓' if y_test[i] == predictions[i] else '✗'
            print(f"Mẫu {i+1}: Actual={actual:7s}, Pred={pred:7s}, "
                  f"Prob={probabilities[i]:.4f} {correct}")

    print("\n" + "="*80)
    print("✅ Demo hoàn thành!")


if __name__ == "__main__":
    demo()


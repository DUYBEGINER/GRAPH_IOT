import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import json
import argparse
from tqdm import tqdm
import glob

# ============================================================================ 
# CONFIGURATION (AUTO-DETECT KAGGLE VS LOCAL)
# ============================================================================ 
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    print("🌍 ENVIRONMENT: KAGGLE DETECTED")
    BASE_DIR = "/kaggle/working"
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_lstm")
else:
    print("💻 ENVIRONMENT: LOCAL DESKTOP DETECTED")
    # Giả sử file này nằm trong GRAPH_IOT/LSTM/
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Trỏ ra thư mục gốc để lấy paths
    BASE_DIR = SCRIPT_DIR
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_lstm')

MODEL_PATH = os.path.join(MODEL_DIR, 'best_lstm_model.keras')
SCALER_PATH = os.path.join(PROCESSED_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(PROCESSED_DIR, 'feature_names.json')
WINDOW_SIZE = 10

print(f"   - Model Path: {MODEL_PATH}")
print(f"   - Scaler Path: {SCALER_PATH}")


class LSTMInference:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, feature_path=FEATURE_NAMES_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_path = feature_path
        self.model = None
        self.scaler = None
        self.feature_names = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load Model, Scaler và Metadata"""
        print("\nLoading model and artifacts...")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model not found at {self.model_path}. Train model first!")

        self.model = tf.keras.models.load_model(self.model_path)
        print("✓ Model loaded.")

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"❌ Scaler not found at {self.scaler_path}. Run preprocessing first!")

        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Scaler loaded.")

        if os.path.exists(self.feature_path):
            with open(self.feature_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"✓ Feature names loaded ({len(self.feature_names)} features).")
        else:
            print("⚠️ Warning: Feature names file not found. Columns order might be incorrect if CSV varies.")

    def preprocess_new_data(self, csv_path):
        """Xử lý dữ liệu mới giống hệt quy trình train"""
        print(f"\nProcessing {os.path.basename(csv_path)}...")
        try:
            df = pd.read_csv(csv_path, low_memory=False, nrows=500000)  # Limit rows for inference safety
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None, None

        # 1. Clean Columns
        if 'Label' in df.columns:
            # Lưu lại label thật để đối chiếu nếu cần
            true_labels = df['Label'].copy()
            df = df.drop(columns=['Label'])

            # Loại bỏ các cột không dùng (giống preprocess)
        cols_to_drop = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port']
        existing_drop = [c for c in cols_to_drop if c in df.columns]
        df.drop(columns=existing_drop, inplace=True)

        # 2. Numeric Conversion
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 3. Handle Infinity
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)

        # 4. Check Feature Consistency
        if self.feature_names:
            # Thêm cột thiếu
            missing_cols = set(self.feature_names) - set(df.columns)
            for c in missing_cols:
                df[c] = 0

            # Sắp xếp đúng thứ tự
            df = df[self.feature_names]

        # 5. Normalize
        X_scaled = self.scaler.transform(df.values)

        return X_scaled, df

    def create_sequences(self, X_scaled):
        """Tạo Sliding Window"""
        Xs = []
        indices = []

        if len(X_scaled) <= WINDOW_SIZE:
            print("⚠️ Data too short for windowing.")
            return np.array([]), []

        # Vectorized implementation for speed could be better, but loop is safe for now
        # Creating a view is faster but keeping it simple
        for i in range(len(X_scaled) - WINDOW_SIZE):
            Xs.append(X_scaled[i: i + WINDOW_SIZE])
            indices.append(i + WINDOW_SIZE)

        return np.array(Xs), indices

    def predict(self, csv_path, threshold=0.5, save_results=True):
        """Hàm dự đoán chính"""
        # 1. Preprocess
        result = self.preprocess_new_data(csv_path)
        if result is None: return
        X_scaled, df_original = result

        if X_scaled is None: return

        # 2. Sequence
        X_seq, indices = self.create_sequences(X_scaled)

        if len(X_seq) == 0:
            return None

        print(f"Running inference on {len(X_seq)} sequences...")

        # 3. Predict
        # batch_size lớn để nhanh
        probs = self.model.predict(X_seq, batch_size=2048, verbose=1)

        # 4. Analyze Results
        preds = (probs > threshold).astype(int).flatten()

        n_attacks = np.sum(preds)
        print(f"\n===== INFERENCE RESULTS: {os.path.basename(csv_path)} =====")
        print(f"Total Flows: {len(preds)}")
        print(f"Benign:      {len(preds) - n_attacks}")
        print(f"Malicious:   {n_attacks} ({(n_attacks / len(preds)) * 100:.2f}%)")

        if n_attacks > 0:
            print("⚠️ ATTACK DETECTED!")

            if save_results:
                output_file = csv_path.replace('.csv', '_predictions.csv')
                # Nếu trên Kaggle, đổi path output về /kaggle/working
                if IS_KAGGLE:
                    output_file = os.path.join(BASE_DIR, f"pred_{os.path.basename(csv_path)}")

                # Tạo df kết quả
                # Chỉ lưu các dòng tương ứng với cuối window
                result_df = df_original.iloc[indices].copy()
                result_df['Attack_Probability'] = probs.flatten()
                result_df['Is_Attack'] = preds

                # Lưu file
                result_df.to_csv(output_file, index=False)
                print(f"📝 Detailed predictions saved to: {output_file}")

        else:
            print("✓ No anomalies detected.")

        return preds, probs


def main():
    parser = argparse.ArgumentParser(description='LSTM Inference')
    parser.add_argument('path', type=str, nargs='?', help='Path to CSV file or Directory')
    parser.add_argument('--threshold', type=float, default=0.5)

    # ====================================================================
    # SỬA LỖI Ở ĐÂY: Thêm args=[] để chạy được trong Notebook/Kaggle
    # ====================================================================
    args = parser.parse_args(args=[])

    # Nếu bạn muốn chỉ định file cụ thể để test thay vì để nó tự tìm,
    # hãy bỏ comment dòng dưới và điền đường dẫn vào:
    # args.path = "/kaggle/input/ids-intrusion-csv/02-14-2018.csv"

    predictor = LSTMInference()

    # Nếu không truyền argument (hoặc args=[]), thử chạy test với file trong dataset (demo)
    if not args.path:
        print("No path provided via arguments. Searching for a demo CSV file...")
        if IS_KAGGLE:
            # Tìm file trong input - Cập nhật đúng folder bạn đang dùng
            possible_dirs = [
                "/kaggle/input/ids-intrusion-csv",
                "/kaggle/input/cicids2018"
            ]
            demo_files = []
            for d in possible_dirs:
                found = glob.glob(os.path.join(d, "*.csv"))
                if found:
                    demo_files.extend(found)

            # Fallback nếu không tìm thấy trong folder chỉ định
            if not demo_files:
                demo_files = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        else:
            # Tìm file local
            demo_files = glob.glob(os.path.join(os.path.dirname(BASE_DIR), "data_IOT", "*.csv"))

        if demo_files:
            # Lấy file đầu tiên tìm thấy để test
            target_path = demo_files[0]
            print(f"Found demo file: {target_path}")
            predictor.predict(target_path)
        else:
            print("❌ No demo files found in /kaggle/input. Please check your dataset.")
    else:
        # Nếu là file
        if os.path.isfile(args.path):
            predictor.predict(args.path, args.threshold)
        # Nếu là folder
        elif os.path.isdir(args.path):
            csv_files = glob.glob(os.path.join(args.path, "*.csv"))
            print(f"Found {len(csv_files)} CSV files in directory.")
            for f in csv_files:
                predictor.predict(f, args.threshold)


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import json
import argparse
from tqdm import tqdm

# ============================================================================ 
# CONFIGURATION
# ============================================================================ 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_lstm_model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'processed_lstm', 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'processed_lstm', 'feature_names.json')
WINDOW_SIZE = 10  # Phải khớp với lúc train

class LSTMInference:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Load Model, Scaler và Metadata"""
        print("Loading model and artifacts...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train model first!")
            
        self.model = tf.keras.models.load_model(self.model_path)
        print("✓ Model loaded.")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Run preprocessing first!")
            
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Scaler loaded.")
        
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, 'r') as f:
                self.feature_names = json.load(f)

    def preprocess_new_data(self, csv_path):
        """Xử lý dữ liệu mới giống hệt quy trình train"""
        print(f"\nProcessing {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        
        # 1. Clean Columns
        # Cố gắng khớp columns với lúc train
        if 'Label' in df.columns:
            df = df.drop(columns=['Label']) # Bỏ nhãn thật nếu có (để test blind) 
            
        # Loại bỏ các cột không dùng
        cols_to_drop = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Label']
        existing_drop = [c for c in cols_to_drop if c in df.columns]
        df.drop(columns=existing_drop, inplace=True)
        
        # 2. Numeric Conversion
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 3. Handle Infinity
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        
        # 4. Check Feature Consistency
        if self.feature_names:
            # Nếu thiếu cột, thêm vào với giá trị 0
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            # Sắp xếp đúng thứ tự
            df = df[self.feature_names]
        
        # 5. Normalize
        X_scaled = self.scaler.transform(df.values)
        
        return X_scaled, df # Trả về cả df gốc để map kết quả

    def create_sequences(self, X_scaled):
        """Tạo Sliding Window cho dữ liệu mới"""
        Xs = []
        indices = [] # Lưu index dòng cuối của window để map về df gốc
        
        if len(X_scaled) <= WINDOW_SIZE:
            print("⚠️ Data too short for windowing.")
            return np.array([]), []
            
        for i in range(len(X_scaled) - WINDOW_SIZE):
            Xs.append(X_scaled[i : i + WINDOW_SIZE])
            indices.append(i + WINDOW_SIZE)
            
        return np.array(Xs), indices

    def predict(self, csv_path, threshold=0.5):
        """Hàm dự đoán chính"""
        # 1. Preprocess
        X_scaled, df_original = self.preprocess_new_data(csv_path)
        
        # 2. Sequence
        X_seq, indices = self.create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            return None
            
        print(f"Predicting on {len(X_seq)} sequences...")
        
        # 3. Predict
        # batch_size lớn để nhanh
        probs = self.model.predict(X_seq, batch_size=1024, verbose=1)
        
        # 4. Analyze Results
        preds = (probs > threshold).astype(int).flatten()
        
        n_attacks = np.sum(preds)
        print(f"\n===== INFERENCE RESULTS =====")
        print(f"Total Flows Checked: {len(preds)}")
        print(f"Benign Flows: {len(preds) - n_attacks}")
        print(f"Malicious Flows: {n_attacks} ({(n_attacks/len(preds))*100:.2f}%)")
        
        if n_attacks > 0:
            print("⚠️ WARNING: ATTACK DETECTED!")
            
            # Map lại vào DataFrame gốc để user biết dòng nào bị tấn công
            # indices lưu vị trí dòng cuối của mỗi window -> đó là dòng bị nghi ngờ
            attack_indices = [indices[i] for i in range(len(preds)) if preds[i] == 1]
            
            # Xuất 5 dòng đầu bị detect
            print("\nSample Malicious Flows (First 5):")
            sample_attacks = df_original.iloc[attack_indices[:5]]
            print(sample_attacks)
            
            # Lưu kết quả
            output_file = csv_path.replace('.csv', '_predictions.csv')
            
            # Tạo df kết quả
            result_df = df_original.iloc[indices].copy()
            result_df['Attack_Probability'] = probs.flatten()
            result_df['Is_Attack'] = preds
            
            # Chỉ lưu những dòng là Attack (hoặc lưu hết tùy nhu cầu)
            result_df.to_csv(output_file, index=False)
            print(f"\nDetailed predictions saved to: {output_file}")
            
        else:
            print("✓ System Secure. No anomalies detected.")

def main():
    parser = argparse.ArgumentParser(description='LSTM Inference for IoT Anomaly Detection')
    parser.add_argument('csv_path', type=str, help='Path to new CSV file to check')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for attack detection (0.0 - 1.0)')
    
    args = parser.parse_args()
    
    predictor = LSTMInference()
    predictor.predict(args.csv_path, args.threshold)

if __name__ == "__main__":
    main()

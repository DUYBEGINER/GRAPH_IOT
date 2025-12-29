import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# CẤU HÌNH
# ============================================================================
MODEL_PATH = "/kaggle/working/models/best_lstm_model.keras"
SCALER_PATH = "/kaggle/working/processed_lstm/scaler.pkl"
FEATURE_PATH = "/kaggle/working/processed_lstm/feature_names.json"
WINDOW_SIZE = 10


class VisualInference:
    def __init__(self):
        print("⏳ Loading model & artifacts...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("❌ Không tìm thấy Model! Hãy chạy file Train trước.")

        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)

        if os.path.exists(FEATURE_PATH):
            with open(FEATURE_PATH, 'r') as f:
                self.features = json.load(f)
        else:
            self.features = None

        print("✅ Model loaded successfully.")

    def process_and_predict(self, csv_path):
        print(f"\nProcessing file: {csv_path.split('/')[-1]}...")

        # 1. Load Data (200k dòng)
        try:
            df = pd.read_csv(csv_path, nrows=200000, low_memory=False)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return

        # 2. Preprocessing Cleanup
        # Bỏ cột Label nếu có
        if 'Label' in df.columns:
            df = df.drop(columns=['Label'])

        # Bỏ cột rác
        cols_to_drop = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # --- FIX LỖI INFINITY Ở ĐÂY ---
        # Chuyển về số
        df = df.apply(pd.to_numeric, errors='coerce')

        # Thay thế Infinity bằng NaN, sau đó thay thế NaN bằng 0
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Đảm bảo kiểu dữ liệu là float32 để tiết kiệm RAM và tránh lỗi
        df = df.astype(np.float32)
        # ------------------------------

        # Đồng bộ cột (Align features)
        if self.features:
            # Thêm cột thiếu
            for c in self.features:
                if c not in df.columns:
                    df[c] = 0
            # Chỉ lấy đúng các cột đã train, theo đúng thứ tự
            df = df[self.features]

        # Normalize
        print("   - Normalizing data...")
        try:
            X_scaled = self.scaler.transform(df.values)
        except ValueError as e:
            print(f"❌ Lỗi Scaler: {e}")
            print("   -> Đang cố gắng sửa lỗi bằng cách reset các giá trị quá lớn...")
            # Fallback: Nếu vẫn lỗi, clip giá trị trong khoảng cho phép của float64
            df = df.clip(lower=-1e10, upper=1e10)
            X_scaled = self.scaler.transform(df.values)

        # Tạo Sequence
        print("   - Creating sequences...")
        X_seq = []
        if len(X_scaled) <= WINDOW_SIZE:
            print("⚠️ File quá ngắn.")
            return

        # Loop nhanh
        for i in range(len(X_scaled) - WINDOW_SIZE):
            X_seq.append(X_scaled[i: i + WINDOW_SIZE])

        X_seq = np.array(X_seq)

        # 3. Predict
        print(f"   - Running inference on {len(X_seq)} sequences...")
        probs = self.model.predict(X_seq, batch_size=2048, verbose=1).flatten()

        # 4. Visualize
        self.visualize(probs, csv_path)

    def visualize(self, probs, filename):
        print("📊 Generating Visualization...")
        sns.set_style("whitegrid")
        preds = (probs > 0.5).astype(int)

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2)

        # Chart 1: Time Series
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(probs, label='Attack Probability', color='royalblue', alpha=0.6, linewidth=0.8)

        attacks = np.where(preds == 1)[0]
        if len(attacks) > 0:
            ax1.scatter(attacks, probs[attacks], color='red', s=3, label='Detected Attack', zorder=5)

        ax1.axhline(0.5, color='orange', linestyle='--', label='Threshold (0.5)')
        ax1.set_title(f'Network Traffic Anomaly: {filename.split("/")[-1]}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Probability')
        ax1.legend(loc='upper right')

        # Chart 2: Histogram
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(probs, bins=50, kde=True, ax=ax2, color='purple')
        ax2.set_title('Confidence Distribution')
        ax2.set_xlabel('Probability Score')

        # Chart 3: Pie Chart
        ax3 = fig.add_subplot(gs[1, 1])
        n_benign = len(preds) - np.sum(preds)
        n_attack = np.sum(preds)

        if n_attack == 0 and n_benign == 0:
            ax3.text(0.5, 0.5, "No Data", ha='center')
        else:
            ax3.pie([n_benign, n_attack],
                    labels=[f'Benign ({n_benign})', f'Attack ({n_attack})'],
                    colors=['#4CAF50', '#F44336'],
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=(0, 0.1))
            ax3.set_title(f'Attack Ratio (Total: {len(preds)})')

        plt.tight_layout()
        plt.show()
        print("✅ Visualization Done.")


# ============================================================================
# MAIN EXECUTION (ĐÃ SỬA ĐỂ CHỌN FILE NHIỀU ATTACK)
# ============================================================================
if __name__ == "__main__":
    predictor = VisualInference()

    # 1. Tìm tất cả file CSV
    search_patterns = [
        "/kaggle/input/cicids2018/CICIDS2018_CSV/*.csv",
        "/kaggle/input/ids-intrusion-csv/*.csv",
        "/kaggle/input/cse-cic-ids2018/*.csv",
        "/kaggle/input/**/*.csv"
    ]

    all_files = []
    for pattern in search_patterns:
        found = glob.glob(pattern, recursive=True)
        all_files.extend(found)

    # Loại bỏ trùng lặp nếu có
    all_files = list(set(all_files))

    if all_files:
        selected_file = None

        # --- DANH SÁCH ƯU TIÊN (Priority List) ---
        # Tìm file chứa DDoS (nhiều Attack nhất) để biểu đồ đẹp
        priority_keywords = [
            "02-20-2018",  # DDoS (LOIC) - Rất nhiều attack
            "02-21-2018",  # DDoS (LOIC/HOIC)
            "02-15-2018",  # DoS (GoldenEye)
            "02-14-2018"  # BruteForce
        ]

        print("🔎 Đang tìm kiếm file chứa nhiều tấn công (DDoS/DoS)...")

        for keyword in priority_keywords:
            for f in all_files:
                if keyword in f:
                    selected_file = f
                    print(f"✅ Đã tìm thấy file ưu tiên: {keyword}")
                    break
            if selected_file: break

        # Nếu không tìm thấy file ưu tiên, lấy file bất kỳ
        if selected_file is None:
            selected_file = all_files[0]
            print(f"⚠️ Không tìm thấy file DDoS, sử dụng file ngẫu nhiên: {selected_file}")

        print(f"👉 Bắt đầu xử lý file: {selected_file}")

        # Chạy dự đoán
        predictor.process_and_predict(selected_file)

    else:
        print("❌ Không tìm thấy bất kỳ file CSV nào trong /kaggle/input.")
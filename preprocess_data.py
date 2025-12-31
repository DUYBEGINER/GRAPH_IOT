"""
CICIDS2018 Data Preprocessing for GNN-based IoT Anomaly Detection
Xử lý dữ liệu CICIDS2018 để chuẩn bị cho GNN model

FEATURE SELECTION STRATEGY:
===========================
Dựa trên phân tích features_summary.txt, script này:

1. LOẠI BỎ các cột không nhất quán giữa các file:
   - Src IP, Dst IP: Chỉ có trong file Thuesday-20-02-2018
   - Src Port: Chỉ có trong file Thuesday-20-02-2018
   - Flow ID: Chỉ có trong file Thuesday-20-02-2018
   - Timestamp: Không cần thiết cho training

2. GIỮ LẠI các features quan trọng cho GNN (75 features):
   - Dst Port: Cổng đích (quan trọng cho phân loại traffic)
   - Protocol: Giao thức mạng
   - Flow features: Duration, Bytes/s, Packets/s, IAT (Inter-Arrival Time)
   - Packet features: Sizes, Lengths (Forward/Backward)
   - TCP Flags: FIN, SYN, RST, PSH, ACK, URG, CWE, ECE
   - Window features: Init Fwd/Bwd Win Bytes
   - Active/Idle time statistics
   - Label: Nhãn phân loại

3. CÁC FEATURES QUAN TRỌNG CHO GNN:
   - Flow statistics giúp GNN học patterns của network traffic
   - TCP flags giúp identify attack patterns
   - Packet size statistics cho behavior analysis
   - IAT features cho timing attack detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not installed. Install for progress bars: pip install tqdm")
    # Fallback: tqdm = lambda x, **kwargs: x nếu không có tqdm
    tqdm = lambda x, **kwargs: x

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
OUTPUT_DIR = r"processed_data"

# MEMORY OPTIMIZATION:
# Full dataset (~16M rows) cần ~20GB RAM
# Nếu gặp MemoryError, set SAMPLE_SIZE để giảm kích thước:
# SAMPLE_SIZE = 1000000  # 1M rows (~2-3GB RAM)
# SAMPLE_SIZE = 500000   # 500k rows (~1-2GB RAM)
SAMPLE_SIZE = None  # None để dùng toàn bộ dữ liệu
RANDOM_STATE = 42

# ============================================================================
# DATA PREPROCESSING CLASS
# ============================================================================

class CICIDS2018Preprocessor:
    """Class để xử lý và chuẩn bị dữ liệu CICIDS2018"""

    def __init__(self, data_dir, output_dir, sample_size=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        # Tạo thư mục output nếu chưa có
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        """Load tất cả các file CSV và merge lại"""
        print("=" * 80)
        print("LOADING CICIDS2018 DATA")
        print("=" * 80)
        csv_files = sorted(Path(self.data_dir).glob("*_TrafficForML_CICFlowMeter.csv"))

        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy file CSV trong {self.data_dir}")
        print(f"Tìm thấy {len(csv_files)} file CSV")
        dfs = []
        total_rows_loaded = 0
        total_rows_filtered = 0

        # Progress bar cho việc đọc files
        csv_files_iter = tqdm(csv_files, desc="📂 Loading CSV files", unit="file") if TQDM_AVAILABLE else csv_files

        for csv_file in csv_files_iter:
            if not TQDM_AVAILABLE:
                print(f"  Đang đọc: {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, low_memory=False, encoding='latin-1')

            initial_rows = len(df)
            total_rows_loaded += initial_rows

            # Lọc bỏ header rows NGAY KHI LOAD để tiết kiệm RAM
            if 'Label' in df.columns:
                mask = df['Label'] != 'Label'
                df = df[mask].copy()

                # THÊM: Convert tất cả cột (trừ Label) về numeric
                if TQDM_AVAILABLE:
                    csv_files_iter.set_postfix({'converting': 'dtypes'})

                for col in df.columns:
                    if col != 'Label':
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                filtered_rows = len(df)
                rows_removed = initial_rows - filtered_rows
                total_rows_filtered += rows_removed

                if TQDM_AVAILABLE:
                    csv_files_iter.set_postfix({'rows': f'{filtered_rows:,}', 'filtered': rows_removed})
                else:
                    if rows_removed > 0:
                        print(f"    ✓ {filtered_rows:,} rows loaded (filtered {rows_removed} header rows)")
                    else:
                        print(f"    ✓ {filtered_rows:,} rows loaded")
            else:
                print(f"    ✓ {initial_rows:,} rows loaded")
            dfs.append(df)

        # Merge all dataframes
        print(f"\nMerging {len(dfs)} dataframes...")
        print(f"Total rows loaded: {total_rows_loaded:,}")
        if total_rows_filtered > 0:
            print(f"Total header rows filtered: {total_rows_filtered:,}")
        data = pd.concat(dfs, ignore_index=True)
        print(f"✓ Total rows after merge: {len(data):,}")
        print(f"✓ Total columns: {len(data.columns)}")

        # Giải phóng bộ nhớ
        del dfs
        import gc
        gc.collect()
        # Sample nếu cần
        if self.sample_size and self.sample_size < len(data):
            print(f"\nSampling {self.sample_size:,} rows...")
            data = data.sample(n=self.sample_size, random_state=RANDOM_STATE).copy()
            gc.collect()

        return data

    def clean_data(self, data):
        """Làm sạch dữ liệu"""
        print("\n" + "=" * 80)
        print("CLEANING DATA")
        print("=" * 80)
        initial_rows = len(data)

        # Xóa các cột không cần thiết và không nhất quán giữa các file
        # Dựa trên features_summary.txt: Loại bỏ Src IP, Dst IP, Src Port, Dst Port, Flow ID, Timestamp
        # Các cột này không cần thiết cho GNN và gây ra sự khác biệt schema giữa các file
        columns_to_drop = [
            'Timestamp',      # Không cần thiết cho training
            'Flow ID',        # Chỉ có trong 1 file
            'Src IP',         # Chỉ có trong 1 file, không cần cho GNN
            'Dst IP',         # Chỉ có trong 1 file, không cần cho GNN
            'Src Port',       # Chỉ có trong 1 file
            'Bwd PSH Flags',
            'Bwd URG Flags',
            'Fwd URG Flags',
            'CWE Flag Count',
        ]
        existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]
        if existing_cols_to_drop:
            print(f"Dropping inconsistent/unnecessary columns: {existing_cols_to_drop}")
            data = data.drop(columns=existing_cols_to_drop)
        print(f"✓ Remaining columns: {len(data.columns)}")

        # Hiển thị danh sách features đang giữ lại (không bao gồm Label)
        feature_cols_preview = [col for col in data.columns if col != 'Label']
        print(f"✓ Features kept: {len(feature_cols_preview)}")
        print(f"  Sample features: {feature_cols_preview[:10]}")

        # Kiểm tra cột Label
        if 'Label' not in data.columns:
            raise ValueError("Không tìm thấy cột 'Label' trong dữ liệu!")

        # Hiển thị các labels (header rows đã được lọc trong load_data)
        unique_labels = sorted(data['Label'].unique())
        print(f"\nCác labels trong dataset: {unique_labels}")
        print(f"Số lượng labels: {len(unique_labels)}")

        # Import garbage collector
        import gc

        # Xử lý missing values (vectorized operation)
        print(f"\nXử lý missing values và infinity...")
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            print(f"  Tìm thấy {missing_count:,} missing values")
            # Fillna inplace để tiết kiệm bộ nhớ
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(0)
            gc.collect()
        else:
            print("  Không có missing values")

        # Xử lý infinity values (vectorized, faster)
        print("  Xử lý infinity values...")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        # Replace inf with 0 using numpy for speed
        cols_iter = tqdm(numeric_cols, desc="🔄 Processing inf values", unit="col") if TQDM_AVAILABLE else numeric_cols
        for col in cols_iter:
            data[col] = np.where(np.isinf(data[col]), 0, data[col])
        gc.collect()

        # Loại bỏ duplicate rows
        print(f"\nKiểm tra duplicates...")
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            print(f"  Tìm thấy {duplicates:,} duplicate rows, đang xóa...")
            data = data.drop_duplicates()
            gc.collect()
        else:
            print("  Không có duplicates")

        final_rows = len(data)
        print(f"\n✓ Rows sau khi clean: {final_rows:,} (loại bỏ {initial_rows - final_rows:,} rows)")

        return data

    def analyze_labels(self, data):
        """Phân tích phân phối labels"""
        print("\n" + "=" * 80)
        print("LABEL DISTRIBUTION")
        print("=" * 80)

        label_counts = data['Label'].value_counts()
        total = len(data)

        print(f"\nTổng số classes: {len(label_counts)}")
        print(f"Tổng số samples: {total:,}\n")

        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"  {label:30s}: {count:8,} ({percentage:5.2f}%)")

        # Tạo binary classification: Benign vs Attack
        print("\n" + "-" * 80)
        print("Binary Classification (Benign vs Attack):")
        print("-" * 80)
        benign_count = (data['Label'] == 'Benign').sum()
        attack_count = total - benign_count
        print(f"  Benign: {benign_count:,} ({benign_count/total*100:.2f}%)")
        print(f"  Attack: {attack_count:,} ({attack_count/total*100:.2f}%)")

        return label_counts

    def create_binary_labels(self, data):
        """Tạo binary labels: 0 = Benign, 1 = Attack"""
        print("\n" + "=" * 80)
        print("CREATING BINARY LABELS")
        print("=" * 80)

        # Lưu original labels
        data['original_label'] = data['Label'].copy()

        # Tạo binary label
        data['binary_label'] = (data['Label'] != 'Benign').astype(int)

        print(f"✓ Binary labels created:")
        print(f"  0 (Benign): {(data['binary_label'] == 0).sum():,}")
        print(f"  1 (Attack): {(data['binary_label'] == 1).sum():,}")

        # Tạo multi-class labels
        print("\nCreating multi-class labels...")
        data['multi_label'] = self.label_encoder.fit_transform(data['Label'])
        print(f"✓ Multi-class labels: {len(self.label_encoder.classes_)} classes")

        return data

    def extract_features(self, data):
        """Trích xuất features (loại bỏ các cột label)"""
        print("\n" + "=" * 80)
        print("EXTRACTING FEATURES")
        print("=" * 80)

        print(f"Total columns in data: {len(data.columns)}")
        print(f"Data types:\n{data.dtypes.value_counts()}")

        # Lấy tất cả cột số, loại trừ labels
        feature_cols = [col for col in data.columns if col not in
                       ['Label', 'original_label', 'binary_label', 'multi_label']]

        print(f"\nColumns after excluding labels: {len(feature_cols)}")

        if len(feature_cols) == 0:
            raise ValueError("Không có features nào sau khi loại bỏ labels!")

        # Chỉ giữ các cột số (int, float)
        numeric_features = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        print(f"Numeric features: {len(numeric_features)}")
        if len(numeric_features) == 0:
            print(f"  Available dtypes in feature_cols: {data[feature_cols].dtypes.value_counts()}")
            raise ValueError("Không có numeric features nào!")

        print(f"Feature names (first 10): {numeric_features[:10]}")

        # Lọc features có variance > 0 (chỉ trên numeric features)
        print("\nLọc features có variance > 0...")
        feature_data = data[numeric_features]

        print(f"Feature data shape: {feature_data.shape}")
        print(f"Feature data dtypes (first 5):")
        for col in numeric_features[:5]:
            print(f"  {col}: {feature_data[col].dtype}")

        variances = feature_data.var(numeric_only=True)
        valid_features = variances[variances > 0].index.tolist()

        print(f"✓ Features với variance > 0: {len(valid_features)}")

        if len(valid_features) == 0:
            print("WARNING: All features have zero variance!")
            print("Sample variances:")
            print(variances.head(10))
            raise ValueError("Tất cả features đều có variance = 0!")

        return data, valid_features

    def normalize_features(self, data, feature_cols):
        """Normalize features sử dụng StandardScaler"""
        print("\n" + "=" * 80)
        print("NORMALIZING FEATURES")
        print("=" * 80)

        if len(feature_cols) == 0:
            raise ValueError("feature_cols is empty! Cannot normalize.")

        print(f"Normalizing {len(feature_cols)} features...")

        if TQDM_AVAILABLE:
            with tqdm(total=2, desc="📊 Normalizing", unit="step") as pbar:
                pbar.set_postfix({'status': 'fitting scaler'})
                data[feature_cols] = self.scaler.fit_transform(data[feature_cols])
                pbar.update(1)

                pbar.set_postfix({'status': 'completed'})
                pbar.update(1)
        else:
            print("Đang fit StandardScaler...")
            data[feature_cols] = self.scaler.fit_transform(data[feature_cols])

        print("✓ Features normalized")
        print(f"  Mean: ~{data[feature_cols].mean().mean():.6f}")
        print(f"  Std: ~{data[feature_cols].std().mean():.6f}")

        return data

    def save_processed_data(self, data, feature_cols):
        """Lưu dữ liệu đã xử lý"""
        print("\n" + "=" * 80)
        print("SAVING PROCESSED DATA")
        print("=" * 80)

        save_steps = [
            ('processed_data.csv', 'CSV file'),
            ('X_features.npy', 'Feature array'),
            ('y_binary.npy', 'Binary labels'),
            ('y_multi.npy', 'Multi labels'),
            ('scaler.pkl', 'Scaler'),
            ('label_encoder.pkl', 'Encoder'),
            ('feature_names.txt', 'Feature names'),
            ('metadata.pkl', 'Metadata')
        ]

        save_iter = tqdm(save_steps, desc="💾 Saving files", unit="file") if TQDM_AVAILABLE else save_steps

        for idx, (filename, description) in enumerate(save_iter):
            if TQDM_AVAILABLE:
                save_iter.set_postfix({'file': filename})

            if idx == 0:  # processed_data.csv
                output_file = os.path.join(self.output_dir, filename)
                if not TQDM_AVAILABLE:
                    print(f"Saving to {output_file}...")
                data.to_csv(output_file, index=False)
                if not TQDM_AVAILABLE:
                    print(f"✓ Saved {len(data):,} rows")

            elif idx == 1:  # X_features.npy
                X = data[feature_cols].values
                np.save(os.path.join(self.output_dir, filename), X)

            elif idx == 2:  # y_binary.npy
                y_binary = data['binary_label'].values
                np.save(os.path.join(self.output_dir, filename), y_binary)

            elif idx == 3:  # y_multi.npy
                y_multi = data['multi_label'].values
                np.save(os.path.join(self.output_dir, filename), y_multi)

            elif idx == 4:  # scaler.pkl
                with open(os.path.join(self.output_dir, filename), 'wb') as f:
                    pickle.dump(self.scaler, f)

            elif idx == 5:  # label_encoder.pkl
                with open(os.path.join(self.output_dir, filename), 'wb') as f:
                    pickle.dump(self.label_encoder, f)

            elif idx == 6:  # feature_names.txt
                with open(os.path.join(self.output_dir, filename), 'w') as f:
                    f.write('\n'.join(feature_cols))

            elif idx == 7:  # metadata.pkl
                metadata = {
                    'n_samples': len(data),
                    'n_features': len(feature_cols),
                    'n_classes': len(self.label_encoder.classes_),
                    'class_names': self.label_encoder.classes_.tolist(),
                    'feature_names': feature_cols
                }
                with open(os.path.join(self.output_dir, filename), 'wb') as f:
                    pickle.dump(metadata, f)

        # Summary
        X = data[feature_cols].values
        y_binary = data['binary_label'].values
        y_multi = data['multi_label'].values

        if not TQDM_AVAILABLE:
            print(f"✓ Saved feature array: {X.shape}")
            print(f"✓ Saved binary labels: {y_binary.shape}")
            print(f"✓ Saved multi-class labels: {y_multi.shape}")
            print("✓ Saved scaler, encoder, and metadata")
        else:
            print(f"\n✓ All files saved successfully!")
            print(f"  - Feature array: {X.shape}")
            print(f"  - Binary labels: {y_binary.shape}")
            print(f"  - Multi-class labels: {y_multi.shape}")

        return X, y_binary, y_multi

    def process_pipeline(self):
        """Pipeline đầy đủ để xử lý dữ liệu"""
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 15 + "CICIDS2018 DATA PREPROCESSING PIPELINE" + " " * 24 + "║")
        print("╚" + "=" * 78 + "╝")
        print("\n")

        # 1. Load data
        data = self.load_data()

        # 2. Clean data
        data = self.clean_data(data)

        # 3. Analyze labels
        self.analyze_labels(data)

        # 4. Create labels
        data = self.create_binary_labels(data)

        # 5. Extract features
        data, feature_cols = self.extract_features(data)

        # 6. Normalize features
        data = self.normalize_features(data, feature_cols)

        # 7. Save processed data
        X, y_binary, y_multi = self.save_processed_data(data, feature_cols)

        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETED!")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Samples: {len(data):,}")
        print(f"Features: {len(feature_cols)}")
        print(f"Classes: {len(self.label_encoder.classes_)}")
        print("=" * 80)

        return data, X, y_binary, y_multi, feature_cols


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""

    # Khởi tạo preprocessor
    preprocessor = CICIDS2018Preprocessor(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        sample_size=SAMPLE_SIZE
    )

    # Chạy preprocessing pipeline
    data, X, y_binary, y_multi, feature_cols = preprocessor.process_pipeline()

    print("\n✓ Dữ liệu đã được xử lý và lưu thành công!")
    print(f"✓ Bạn có thể sử dụng dữ liệu từ thư mục: {OUTPUT_DIR}")

    return data, X, y_binary, y_multi, feature_cols


if __name__ == "__main__":
    data, X, y_binary, y_multi, feature_cols = main()


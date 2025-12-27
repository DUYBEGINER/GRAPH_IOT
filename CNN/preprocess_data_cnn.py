"""
======================================================================================
TIỀN XỬ LÝ DATASET CICIDS2018 CHO MÔ HÌNH CNN - PHÁT HIỆN LƯU LƯỢNG MẠNG IOT BẤT THƯỜNG
======================================================================================

Script này thực hiện các bước tiền xử lý dữ liệu:
1. Đọc từng file CSV theo chunks để tối ưu bộ nhớ
2. Loại bỏ các cột không cần thiết (IP, Port, Timestamp, Flow ID)
3. Xử lý missing values, NaN, Inf
4. Loại bỏ các hàng trùng lặp
5. Chuyển đổi nhãn sang dạng binary (Benign=0, Attack=1)
6. Chuẩn hóa dữ liệu bằng StandardScaler
7. Lưu dữ liệu đã xử lý sang định dạng nhanh (parquet/npy)

Có thể chạy trên cả Kaggle và Local
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import gc
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# THƯ VIỆN CHUẨN HÓA VÀ XỬ LÝ DỮ LIỆU
# ============================================================================
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Kiểm tra môi trường chạy (Kaggle hoặc Local)
IS_KAGGLE = os.path.exists('/kaggle/input')

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm không có sẵn. Cài đặt bằng: pip install tqdm")
    tqdm = lambda x, **kwargs: x

# ============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================================
if IS_KAGGLE:
    # Đường dẫn trên Kaggle - thay đổi theo dataset của bạn
    DATA_DIR = "/kaggle/input/cicids2018"  # Thay đổi nếu tên dataset khác
    OUTPUT_DIR = "/kaggle/working/processed_data_cnn"
    print(" Đang chạy trên KAGGLE")
else:
    # Đường dẫn Local
    DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
    OUTPUT_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\processed_data_cnn"
    print(" Đang chạy trên LOCAL")

# ============================================================================
# CẤU HÌNH XỬ LÝ DỮ LIỆU
# ============================================================================

# Kích thước chunk khi đọc CSV (điều chỉnh theo RAM của máy)
CHUNK_SIZE = 300000  # 300k rows mỗi chunk

# Random state để tái tạo kết quả
RANDOM_STATE = 42

# Loại scaler: 'standard' (StandardScaler) hoặc 'minmax' (MinMaxScaler)
SCALER_TYPE = 'standard'

# ============================================================================
# CẤU HÌNH SAMPLE CÂN BẰNG
# ============================================================================
# Tổng số mẫu mong muốn (train + val + test)
TOTAL_SAMPLES = 4000000  # 3 triệu mẫu

# Tỷ lệ phần trăm cho mỗi class
BENIGN_RATIO = 0.50  # 70% Benign = 2,100,000 mẫu
ATTACK_RATIO = 0.50  # 30% Attack = 900,000 mẫu

# Tính số lượng mẫu cho mỗi class
TARGET_BENIGN = int(TOTAL_SAMPLES * BENIGN_RATIO)  # 2,100,000
TARGET_ATTACK = int(TOTAL_SAMPLES * ATTACK_RATIO)  # 900,000

# ============================================================================
# DANH SÁCH CÁC CỘT CẦN LOẠI BỎ
# ============================================================================

# Các cột không cần thiết cho việc huấn luyện CNN
COLUMNS_TO_DROP = [
    # Thông tin định danh - không mang tính tổng quát
    'Flow ID',          # ID duy nhất cho mỗi flow
    'Src IP',           # IP nguồn
    'Dst IP',           # IP đích
    'Src Port',         # Port nguồn
    'Timestamp',        # Thời gian - không liên quan đến pattern

    # Các cột flag không mang nhiều thông tin
    # 'Bwd PSH Flags',    # Hiếm khi có giá trị khác 0
    # 'Bwd URG Flags',    # Hiếm khi có giá trị khác 0
    # 'Fwd URG Flags',    # Hiếm khi có giá trị khác 0
]

# Cột nhãn
LABEL_COLUMN = 'Label'

# ============================================================================
# CLASS XỬ LÝ DỮ LIỆU CHO CNN
# ============================================================================

class CICIDS2018_CNN_Preprocessor:
    """
    Class xử lý dữ liệu CICIDS2018 cho mô hình CNN phát hiện bất thường

    Các bước xử lý:
    1. Đọc dữ liệu theo chunks
    2. Loại bỏ cột không cần thiết
    3. Xử lý giá trị thiếu, NaN, Inf
    4. Loại bỏ duplicate
    5. Chuyển đổi nhãn sang binary
    6. Chuẩn hóa features
    7. Lưu dữ liệu đã xử lý
    """

    def __init__(self, data_dir, output_dir, chunk_size=CHUNK_SIZE,
                 scaler_type=SCALER_TYPE, target_benign=TARGET_BENIGN,
                 target_attack=TARGET_ATTACK):
        """
        Khởi tạo preprocessor

        Args:
            data_dir: Đường dẫn thư mục chứa file CSV
            output_dir: Đường dẫn thư mục lưu kết quả
            chunk_size: Số dòng mỗi chunk khi đọc CSV
            scaler_type: Loại scaler ('standard' hoặc 'minmax')
            target_benign: Số lượng mẫu Benign mong muốn
            target_attack: Số lượng mẫu Attack mong muốn
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.scaler_type = scaler_type
        self.target_benign = target_benign
        self.target_attack = target_attack

        # Khởi tạo scaler
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        # Tạo thư mục output nếu chưa có
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Thống kê
        self.stats = {
            'total_rows_read': 0,
            'rows_after_cleaning': 0,
            'duplicates_removed': 0,
            'nan_inf_replaced': 0,
            'benign_count': 0,
            'attack_count': 0,
            'feature_count': 0,
            'processing_time': 0
        }

        # Lưu tên các features
        self.feature_names = None

    def _get_csv_files(self):
        """Lấy danh sách các file CSV trong thư mục data"""
        csv_files = list(self.data_dir.glob("*_TrafficForML_CICFlowMeter.csv"))
        if not csv_files:
            # Thử pattern khác cho Kaggle
            csv_files = list(self.data_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy file CSV trong {self.data_dir}")

        print(f"\n📂 Tìm thấy {len(csv_files)} file CSV:")
        for f in sorted(csv_files):
            print(f"   - {f.name}")
        return sorted(csv_files)

    def _clean_column_names(self, df):
        """Chuẩn hóa tên cột (loại bỏ khoảng trắng thừa)"""
        df.columns = df.columns.str.strip()
        return df

    def _drop_unnecessary_columns(self, df):
        """Loại bỏ các cột không cần thiết cho huấn luyện"""
        columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]

        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        return df

    def _convert_to_numeric(self, df):
        """Chuyển đổi các cột về dạng số"""
        # Lấy tất cả cột trừ Label
        feature_cols = [col for col in df.columns if col != LABEL_COLUMN]

        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _handle_nan_inf(self, df):
        """Xử lý giá trị NaN và Infinity"""
        feature_cols = [col for col in df.columns if col != LABEL_COLUMN]

        # Đếm số lượng NaN và Inf trước khi xử lý
        nan_count = df[feature_cols].isna().sum().sum()
        inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()

        self.stats['nan_inf_replaced'] += nan_count + inf_count

        # Thay thế Infinity bằng NaN trước
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Thay thế NaN bằng 0 (hoặc có thể dùng median/mean)
        df[feature_cols] = df[feature_cols].fillna(0)

        return df

    def _remove_duplicates(self, df):
        """Loại bỏ các hàng trùng lặp"""
        rows_before = len(df)
        df = df.drop_duplicates()
        rows_after = len(df)

        self.stats['duplicates_removed'] += (rows_before - rows_after)

        return df

    def _convert_to_binary_label(self, df):
        """
        Chuyển đổi nhãn sang dạng binary:
        - Benign -> 0 (lưu lượng bình thường)
        - Tất cả các loại tấn công khác -> 1 (lưu lượng bất thường)
        """
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{LABEL_COLUMN}' trong dữ liệu")

        # Chuẩn hóa nhãn (loại bỏ khoảng trắng, lowercase)
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()

        # Loại bỏ các hàng có nhãn là 'label' (header bị lẫn vào data)
        df = df[df[LABEL_COLUMN] != 'label']

        # Chuyển đổi sang binary: Benign=0, Attack=1
        df['binary_label'] = (df[LABEL_COLUMN] != 'benign').astype(int)

        # Đếm số lượng mỗi class
        benign_count = (df['binary_label'] == 0).sum()
        attack_count = (df['binary_label'] == 1).sum()

        self.stats['benign_count'] += benign_count
        self.stats['attack_count'] += attack_count

        # Xóa cột Label gốc, giữ lại binary_label
        df = df.drop(columns=[LABEL_COLUMN])

        return df

    def _process_single_file(self, csv_file):
        """
        Xử lý một file CSV theo chunks

        Args:
            csv_file: Đường dẫn file CSV

        Returns:
            DataFrame đã được xử lý
        """
        print(f"\n📄 Đang xử lý: {csv_file.name}")

        processed_chunks = []
        chunk_iterator = pd.read_csv(csv_file, chunksize=self.chunk_size,
                                     low_memory=False, encoding='utf-8')

        # Progress bar cho chunks
        if TQDM_AVAILABLE:
            # Ước tính số chunks dựa trên file size
            file_size = csv_file.stat().st_size
            estimated_chunks = max(1, file_size // (self.chunk_size * 500))  # Ước tính
            chunk_iterator = tqdm(chunk_iterator, desc="   Chunks",
                                  total=estimated_chunks, unit="chunk")

        for chunk in chunk_iterator:
            self.stats['total_rows_read'] += len(chunk)

            # Bước 1: Chuẩn hóa tên cột
            chunk = self._clean_column_names(chunk)

            # Bước 2: Loại bỏ cột không cần thiết
            chunk = self._drop_unnecessary_columns(chunk)

            # Bước 3: Chuyển đổi sang dạng số
            chunk = self._convert_to_numeric(chunk)

            # Bước 4: Xử lý NaN và Inf
            chunk = self._handle_nan_inf(chunk)

            # Bước 5: Chuyển đổi nhãn sang binary
            chunk = self._convert_to_binary_label(chunk)

            processed_chunks.append(chunk)

            # Giải phóng bộ nhớ
            gc.collect()

        # Gộp các chunks lại
        if processed_chunks:
            df = pd.concat(processed_chunks, ignore_index=True)
            del processed_chunks
            gc.collect()
            return df

        return None

    def process_all_files(self):
        """
        Xử lý tất cả các file CSV và gộp lại

        Returns:
            DataFrame đã xử lý hoàn chỉnh
        """
        start_time = datetime.now()
        print("\n" + "="*80)
        print("🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU CICIDS2018 CHO CNN")
        print("="*80)

        csv_files = self._get_csv_files()

        all_dataframes = []

        # Xử lý từng file
        for csv_file in csv_files:
            df = self._process_single_file(csv_file)
            if df is not None:
                all_dataframes.append(df)
                print(f"   ✅ Đã xử lý: {len(df):,} mẫu")

        # Gộp tất cả lại
        print("\n" + "-"*80)
        print("📊 ĐANG GỘP VÀ XỬ LÝ CUỐI CÙNG...")

        df_combined = pd.concat(all_dataframes, ignore_index=True)
        del all_dataframes
        gc.collect()

        print(f"   Tổng số mẫu sau khi gộp: {len(df_combined):,}")

        # Loại bỏ duplicate trên toàn bộ dataset
        print("   Đang loại bỏ duplicate...")
        df_combined = self._remove_duplicates(df_combined)
        print(f"   Số mẫu sau khi loại duplicate: {len(df_combined):,}")

        # Cập nhật thống kê
        self.stats['rows_after_cleaning'] = len(df_combined)
        self.stats['feature_count'] = len(df_combined.columns) - 1  # Trừ cột label

        # Lưu tên features
        self.feature_names = [col for col in df_combined.columns if col != 'binary_label']

        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()

        return df_combined

    def balanced_sample(self, df):
        """
        Sample dữ liệu với số lượng cân bằng theo target đã định

        Lấy chính xác:
        - TARGET_BENIGN mẫu Benign (2,100,000)
        - TARGET_ATTACK mẫu Attack (900,000)

        Args:
            df: DataFrame đã clean

        Returns:
            DataFrame đã được sample cân bằng
        """
        print("\n" + "="*80)
        print("⚖️ ĐANG SAMPLE CÂN BẰNG DỮ LIỆU")
        print("="*80)

        # Tách theo class
        df_benign = df[df['binary_label'] == 0]
        df_attack = df[df['binary_label'] == 1]

        n_benign = len(df_benign)
        n_attack = len(df_attack)

        print(f"\n   Dữ liệu gốc (sau khi clean):")
        print(f"   - Benign: {n_benign:,}")
        print(f"   - Attack: {n_attack:,}")
        print(f"   - Tổng: {n_benign + n_attack:,}")

        print(f"\n   Target mong muốn:")
        print(f"   - Benign: {self.target_benign:,} ({BENIGN_RATIO*100:.0f}%)")
        print(f"   - Attack: {self.target_attack:,} ({ATTACK_RATIO*100:.0f}%)")
        print(f"   - Tổng: {self.target_benign + self.target_attack:,}")

        # Kiểm tra và điều chỉnh nếu không đủ mẫu
        actual_benign = min(self.target_benign, n_benign)
        actual_attack = min(self.target_attack, n_attack)

        if actual_benign < self.target_benign:
            print(f"\n   ⚠️ Không đủ Benign! Chỉ có {n_benign:,}, cần {self.target_benign:,}")
        if actual_attack < self.target_attack:
            print(f"\n   ⚠️ Không đủ Attack! Chỉ có {n_attack:,}, cần {self.target_attack:,}")

        # Random sample từ mỗi class
        print(f"\n   Đang sample...")
        df_benign_sampled = df_benign.sample(n=actual_benign, random_state=RANDOM_STATE)
        df_attack_sampled = df_attack.sample(n=actual_attack, random_state=RANDOM_STATE)

        # Gộp lại và shuffle
        df_balanced = pd.concat([df_benign_sampled, df_attack_sampled], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        # Thống kê kết quả
        final_benign = (df_balanced['binary_label'] == 0).sum()
        final_attack = (df_balanced['binary_label'] == 1).sum()
        total = len(df_balanced)

        print(f"\n   ✅ Kết quả sau khi sample:")
        print(f"   - Benign: {final_benign:,} ({final_benign/total*100:.1f}%)")
        print(f"   - Attack: {final_attack:,} ({final_attack/total*100:.1f}%)")
        print(f"   - Tổng: {total:,}")
        print(f"   - Tỷ lệ Benign:Attack = {final_benign/final_attack:.2f}:1")

        # Cập nhật stats
        self.stats['benign_count'] = final_benign
        self.stats['attack_count'] = final_attack
        self.stats['rows_after_cleaning'] = total

        return df_balanced

    def normalize_features(self, df):
        """
        Chuẩn hóa các features bằng scaler

        Args:
            df: DataFrame chứa features và label

        Returns:
            X_normalized: Features đã chuẩn hóa
            y: Labels
        """
        print("\n🔄 ĐANG CHUẨN HÓA DỮ LIỆU...")

        # Tách features và label
        X = df.drop(columns=['binary_label']).values
        y = df['binary_label'].values

        # Chuẩn hóa features
        X_normalized = self.scaler.fit_transform(X)

        print(f"   Scaler type: {self.scaler_type}")
        print(f"   Shape X: {X_normalized.shape}")
        print(f"   Shape y: {y.shape}")

        return X_normalized, y

    def reshape_for_cnn(self, X):
        """
        Reshape dữ liệu cho CNN 1D

        CNN 1D yêu cầu input shape: (samples, features, channels)
        Trong trường hợp này: (samples, n_features, 1)

        Args:
            X: Features đã chuẩn hóa, shape (samples, features)

        Returns:
            X_reshaped: Shape (samples, features, 1)
        """
        print("\n🔄 ĐANG RESHAPE DỮ LIỆU CHO CNN...")

        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

        print(f"   Shape sau reshape: {X_reshaped.shape}")

        return X_reshaped

    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Chia dữ liệu thành train/val/test sets

        Args:
            X: Features
            y: Labels
            test_size: Tỷ lệ test set
            val_size: Tỷ lệ validation set (từ train)

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\n📊 ĐANG CHIA DỮ LIỆU TRAIN/VAL/TEST...")

        # Chia train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        # Chia train / val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
        )

        print(f"   Train set: {X_train.shape[0]:,} mẫu")
        print(f"   Val set:   {X_val.shape[0]:,} mẫu")
        print(f"   Test set:  {X_test.shape[0]:,} mẫu")

        # Thống kê phân bố class
        print(f"\n   Phân bố Train - Benign: {(y_train==0).sum():,}, Attack: {(y_train==1).sum():,}")
        print(f"   Phân bố Val   - Benign: {(y_val==0).sum():,}, Attack: {(y_val==1).sum():,}")
        print(f"   Phân bố Test  - Benign: {(y_test==0).sum():,}, Attack: {(y_test==1).sum():,}")

        return X_train, X_val, X_test, y_train, y_val, y_test


    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Lưu dữ liệu đã xử lý sang định dạng nhanh

        Lưu thành các file:
        - X_train.npy, X_val.npy, X_test.npy
        - y_train.npy, y_val.npy, y_test.npy
        - scaler.pkl
        - metadata.json
        """
        print("\n ĐANG LƯU DỮ LIỆU ĐÃ XỬ LÝ...")

        # Lưu numpy arrays
        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'X_val.npy', X_val)
        np.save(self.output_dir / 'X_test.npy', X_test)
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'y_val.npy', y_val)
        np.save(self.output_dir / 'y_test.npy', y_test)

        print(f"   ✅ Đã lưu X_train.npy: {X_train.shape}")
        print(f"   ✅ Đã lưu X_val.npy: {X_val.shape}")
        print(f"   ✅ Đã lưu X_test.npy: {X_test.shape}")
        print(f"   ✅ Đã lưu y_train.npy: {y_train.shape}")
        print(f"   ✅ Đã lưu y_val.npy: {y_val.shape}")
        print(f"   ✅ Đã lưu y_test.npy: {y_test.shape}")

        # Lưu scaler
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✅ Đã lưu scaler.pkl")

        # Lưu feature names
        with open(self.output_dir / 'feature_names.txt', 'w') as f:
            for name in self.feature_names:
                f.write(name + '\n')
        print(f"   ✅ Đã lưu feature_names.txt")

        # Chuyển đổi stats sang kiểu Python native (để tránh lỗi JSON với numpy.int64)
        stats_native = {}
        for key, value in self.stats.items():
            if hasattr(value, 'item'):  # Kiểm tra nếu là numpy type
                stats_native[key] = value.item()
            elif isinstance(value, (np.integer, np.floating)):
                stats_native[key] = int(value) if isinstance(value, np.integer) else float(value)
            else:
                stats_native[key] = value

        # Lưu metadata
        metadata = {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'input_shape': [int(x) for x in X_train.shape[1:]],
            'scaler_type': self.scaler_type,
            'stats': stats_native,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"   ✅ Đã lưu metadata.json")

        print(f"\n📁 Tất cả file được lưu tại: {self.output_dir}")

    def print_summary(self):
        """In tóm tắt quá trình xử lý"""
        print("\n" + "="*80)
        print("📊 TÓM TẮT XỬ LÝ DỮ LIỆU")
        print("="*80)
        print(f"   Tổng số dòng đọc được:     {self.stats['total_rows_read']:,}")
        print(f"   Số dòng sau khi xử lý:     {self.stats['rows_after_cleaning']:,}")
        print(f"   Số duplicate đã loại:      {self.stats['duplicates_removed']:,}")
        print(f"   Số NaN/Inf đã thay thế:    {self.stats['nan_inf_replaced']:,}")
        print(f"   Số features:               {self.stats['feature_count']}")
        print(f"   Số mẫu Benign:             {self.stats['benign_count']:,}")
        print(f"   Số mẫu Attack:             {self.stats['attack_count']:,}")
        print(f"   Thời gian xử lý:           {self.stats['processing_time']:.2f} giây")
        print("="*80)


def main():
    """Hàm chính để chạy preprocessing"""

    print("\n" + "="*80)
    print("🔧 TIỀN XỬ LÝ DỮ LIỆU CICIDS2018 CHO MÔ HÌNH CNN")
    print("   Phát hiện lưu lượng mạng IoT bất thường")
    print("="*80)

    print(f"\n📋 CẤU HÌNH:")
    print(f"   - Tổng mẫu mong muốn: {TOTAL_SAMPLES:,}")
    print(f"   - Benign: {TARGET_BENIGN:,} ({BENIGN_RATIO*100:.0f}%)")
    print(f"   - Attack: {TARGET_ATTACK:,} ({ATTACK_RATIO*100:.0f}%)")

    # Khởi tạo preprocessor
    preprocessor = CICIDS2018_CNN_Preprocessor(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        chunk_size=CHUNK_SIZE,
        scaler_type=SCALER_TYPE,
        target_benign=TARGET_BENIGN,
        target_attack=TARGET_ATTACK
    )

    # Bước 1: Xử lý tất cả các file CSV (clean data)
    df = preprocessor.process_all_files()

    # Bước 2: SAMPLE CÂN BẰNG TRƯỚC KHI CHIA
    # Điều này đảm bảo train/val/test đều có tỷ lệ 70-30
    df = preprocessor.balanced_sample(df)

    # Bước 3: Chuẩn hóa features
    X, y = preprocessor.normalize_features(df)

    # Giải phóng bộ nhớ của DataFrame
    del df
    gc.collect()

    # Bước 4: Reshape cho CNN
    X = preprocessor.reshape_for_cnn(X)

    # Bước 5: Chia dữ liệu (stratify để giữ tỷ lệ 70-30 trong tất cả các tập)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    # Giải phóng bộ nhớ
    del X, y
    gc.collect()

    # Bước 6: Lưu dữ liệu
    preprocessor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

    # In tóm tắt
    preprocessor.print_summary()

    print("\n✅ HOÀN THÀNH! Dữ liệu đã sẵn sàng cho việc huấn luyện CNN.")

    return preprocessor


if __name__ == "__main__":
    preprocessor = main()


"""
======================================================================================
BƯỚC 2: CHUẨN BỊ DỮ LIỆU TRAINING CHO CNN - CÂN BẰNG VÀ CHIA TRAIN/VAL/TEST
======================================================================================

Script này thực hiện:
1. Đọc dữ liệu đã clean từ step1
2. Cân bằng số lượng nhãn (70% Benign, 30% Attack hoặc tỷ lệ tùy chỉnh)
3. Áp dụng Log Transform: log_e(1+x)
4. Chuẩn hóa bằng StandardScaler
5. Reshape cho CNN 1D
6. Chia train/val/test với stratify để giữ tỷ lệ
7. Lưu dữ liệu đã xử lý để train

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Kiểm tra môi trường chạy (Kaggle hoặc Local)
IS_KAGGLE = os.path.exists('/kaggle/input')

# ============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================================
if IS_KAGGLE:
    CLEANED_DATA_DIR = "/kaggle/working/cleaned_data"
    OUTPUT_DIR = "/kaggle/working/training_data"
    print("🌐 Đang chạy trên KAGGLE")
else:
    CLEANED_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\cleaned_data"
    OUTPUT_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\training_data"
    print("💻 Đang chạy trên LOCAL")

# ============================================================================
# CẤU HÌNH CÂN BẰNG DỮ LIỆU
# ============================================================================

# Random state để tái tạo kết quả
RANDOM_STATE = 42

# Tổng số mẫu mong muốn (train + val + test)
TOTAL_SAMPLES = 3000000  # 3 triệu mẫu

# Tỷ lệ phần trăm cho mỗi class
BENIGN_RATIO = 0.70  # 70% Benign
ATTACK_RATIO = 0.30  # 30% Attack

# Tính số lượng mẫu cho mỗi class
TARGET_BENIGN = int(TOTAL_SAMPLES * BENIGN_RATIO)  # 2,100,000
TARGET_ATTACK = int(TOTAL_SAMPLES * ATTACK_RATIO)  # 900,000

# Tỷ lệ chia train/val/test
TEST_SIZE = 0.20   # 20% cho test
VAL_SIZE = 0.10    # 10% cho validation (từ tổng)
# Train sẽ là 70%

# ============================================================================
# CLASS CHUẨN BỊ DỮ LIỆU TRAINING
# ============================================================================

class TrainingDataPreparer:
    """
    Class chuẩn bị dữ liệu training cho CNN

    Các bước:
    1. Đọc dữ liệu đã clean
    2. Cân bằng dữ liệu theo tỷ lệ mong muốn
    3. Áp dụng log transform: log_e(1+x)
    4. Chuẩn hóa bằng StandardScaler
    5. Reshape cho CNN
    6. Chia train/val/test
    7. Lưu dữ liệu
    """

    def __init__(self, cleaned_data_dir, output_dir,
                 total_samples=TOTAL_SAMPLES,
                 benign_ratio=BENIGN_RATIO,
                 attack_ratio=ATTACK_RATIO,
                 test_size=TEST_SIZE,
                 val_size=VAL_SIZE):
        """
        Khởi tạo preparer

        Args:
            cleaned_data_dir: Đường dẫn thư mục chứa dữ liệu đã clean
            output_dir: Đường dẫn thư mục lưu kết quả
            total_samples: Tổng số mẫu mong muốn
            benign_ratio: Tỷ lệ Benign (0-1)
            attack_ratio: Tỷ lệ Attack (0-1)
            test_size: Tỷ lệ test set
            val_size: Tỷ lệ validation set
        """
        self.cleaned_data_dir = Path(cleaned_data_dir)
        self.output_dir = Path(output_dir)
        self.total_samples = total_samples
        self.benign_ratio = benign_ratio
        self.attack_ratio = attack_ratio
        self.test_size = test_size
        self.val_size = val_size

        # Tính target cho mỗi class
        self.target_benign = int(total_samples * benign_ratio)
        self.target_attack = int(total_samples * attack_ratio)

        # Khởi tạo scaler
        self.scaler = StandardScaler()

        # Tạo thư mục output
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lưu tên features
        self.feature_names = None

        # Thống kê
        self.stats = {
            'original_benign': 0,
            'original_attack': 0,
            'sampled_benign': 0,
            'sampled_attack': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'n_features': 0
        }

    def load_cleaned_data(self):
        """Đọc dữ liệu đã clean từ step1"""
        print("\n" + "="*80)
        print("📂 ĐANG ĐỌC DỮ LIỆU ĐÃ CLEAN...")
        print("="*80)

        parquet_path = self.cleaned_data_dir / 'cleaned_data.parquet'

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file {parquet_path}\n"
                f"Hãy chạy step1_clean_data.py trước!"
            )

        df = pd.read_parquet(parquet_path)

        # Đọc feature names
        feature_names_path = self.cleaned_data_dir / 'feature_names.txt'
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        else:
            self.feature_names = [col for col in df.columns if col != 'binary_label']

        # Thống kê
        self.stats['original_benign'] = int((df['binary_label'] == 0).sum())
        self.stats['original_attack'] = int((df['binary_label'] == 1).sum())
        self.stats['n_features'] = len(self.feature_names)

        print(f"   ✅ Đã đọc: {len(df):,} mẫu")
        print(f"   📊 Phân bố gốc:")
        print(f"      - Benign: {self.stats['original_benign']:,} ({self.stats['original_benign']/len(df)*100:.1f}%)")
        print(f"      - Attack: {self.stats['original_attack']:,} ({self.stats['original_attack']/len(df)*100:.1f}%)")
        print(f"   📋 Số features: {self.stats['n_features']}")

        return df

    def balanced_sample(self, df):
        """
        Sample dữ liệu với tỷ lệ cân bằng mong muốn

        Chiến lược:
        - Nếu có đủ mẫu: lấy đúng số lượng target
        - Nếu không đủ Attack: giảm Benign tương ứng để giữ tỷ lệ
        - Nếu không đủ cả hai: lấy tối đa có thể với tỷ lệ đúng
        """
        print("\n" + "="*80)
        print("⚖️ ĐANG CÂN BẰNG DỮ LIỆU...")
        print("="*80)

        # Tách theo class
        df_benign = df[df['binary_label'] == 0]
        df_attack = df[df['binary_label'] == 1]

        n_benign = len(df_benign)
        n_attack = len(df_attack)

        print(f"\n   🎯 Target mong muốn:")
        print(f"      - Tổng: {self.total_samples:,}")
        print(f"      - Benign: {self.target_benign:,} ({self.benign_ratio*100:.0f}%)")
        print(f"      - Attack: {self.target_attack:,} ({self.attack_ratio*100:.0f}%)")

        # Xác định số lượng thực tế có thể lấy
        # Ưu tiên giữ đúng tỷ lệ
        actual_attack = min(self.target_attack, n_attack)
        # Tính Benign dựa trên Attack thực tế để giữ tỷ lệ
        actual_benign = int(actual_attack * (self.benign_ratio / self.attack_ratio))
        actual_benign = min(actual_benign, n_benign)

        # Nếu Benign bị giới hạn, điều chỉnh Attack
        if actual_benign < int(actual_attack * (self.benign_ratio / self.attack_ratio)):
            actual_attack = int(actual_benign * (self.attack_ratio / self.benign_ratio))

        print(f"\n   📊 Số lượng thực tế sẽ lấy:")
        print(f"      - Benign: {actual_benign:,}")
        print(f"      - Attack: {actual_attack:,}")
        print(f"      - Tổng: {actual_benign + actual_attack:,}")
        print(f"      - Tỷ lệ thực tế: {actual_benign/(actual_benign+actual_attack)*100:.1f}% - {actual_attack/(actual_benign+actual_attack)*100:.1f}%")

        if actual_benign < self.target_benign or actual_attack < self.target_attack:
            print(f"\n   ⚠️ Không đủ mẫu để đạt target!")
            print(f"      Có sẵn: Benign={n_benign:,}, Attack={n_attack:,}")

        # Random sample từ mỗi class
        print(f"\n   🔄 Đang sample...")

        # Sử dụng random sampling
        df_benign_sampled = df_benign.sample(n=actual_benign, random_state=RANDOM_STATE)
        df_attack_sampled = df_attack.sample(n=actual_attack, random_state=RANDOM_STATE)

        # Gộp lại và shuffle
        df_balanced = pd.concat([df_benign_sampled, df_attack_sampled], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        # Cập nhật stats
        self.stats['sampled_benign'] = actual_benign
        self.stats['sampled_attack'] = actual_attack

        print(f"\n   ✅ Kết quả sau khi cân bằng:")
        print(f"      - Benign: {actual_benign:,} ({actual_benign/(actual_benign+actual_attack)*100:.1f}%)")
        print(f"      - Attack: {actual_attack:,} ({actual_attack/(actual_benign+actual_attack)*100:.1f}%)")
        print(f"      - Tổng: {len(df_balanced):,}")

        # Giải phóng bộ nhớ
        del df_benign, df_attack, df_benign_sampled, df_attack_sampled
        gc.collect()

        return df_balanced

    def apply_log_transform(self, X):
        """
        Áp dụng Log Transform: log_e(1+x)

        Lưu ý: log(1+x) giúp:
        - Giảm skewness của dữ liệu
        - Xử lý các giá trị lớn
        - Bảo toàn giá trị 0 (log(1+0) = 0)
        """
        print("\n🔢 ĐANG ÁP DỤNG LOG TRANSFORM: log_e(1+x)...")

        # Đảm bảo không có giá trị âm (log không xác định cho số âm)
        # Với dữ liệu network flow, các giá trị thường >= 0
        # Nếu có giá trị âm, ta shift để min = 0
        min_val = X.min()
        if min_val < 0:
            print(f"   ⚠️ Phát hiện giá trị âm (min={min_val:.4f}), đang shift...")
            X = X - min_val  # Shift để min = 0

        # Áp dụng log(1+x)
        X_log = np.log1p(X)  # log1p(x) = log(1+x), ổn định hơn với x nhỏ

        print(f"   ✅ Log transform hoàn tất")
        print(f"      Range trước: [{X.min():.4f}, {X.max():.4f}]")
        print(f"      Range sau:   [{X_log.min():.4f}, {X_log.max():.4f}]")

        return X_log

    def normalize_features(self, X):
        """
        Chuẩn hóa features bằng StandardScaler
        """
        print("\n📐 ĐANG CHUẨN HÓA BẰNG STANDARDSCALER...")

        X_normalized = self.scaler.fit_transform(X)

        print(f"   ✅ StandardScaler hoàn tất")
        print(f"      Mean: {X_normalized.mean():.6f}")
        print(f"      Std:  {X_normalized.std():.6f}")

        return X_normalized

    def reshape_for_cnn(self, X):
        """
        Reshape dữ liệu cho CNN 1D
        CNN 1D yêu cầu input shape: (samples, features, channels)
        """
        print("\n🔄 ĐANG RESHAPE CHO CNN 1D...")

        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

        print(f"   ✅ Shape: {X.shape} -> {X_reshaped.shape}")
        print(f"      (samples, features, channels)")

        return X_reshaped

    def split_data(self, X, y):
        """
        Chia dữ liệu thành train/val/test

        Thêm validation: Train 70%, Val 10%, Test 20%

        Sử dụng stratify để giữ tỷ lệ class trong tất cả các tập
        """
        print("\n📊 ĐANG CHIA DỮ LIỆU TRAIN/VAL/TEST...")

        # Bước 1: Chia train+val / test (80/20)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=RANDOM_STATE,
            stratify=y  # Giữ tỷ lệ class
        )

        # Bước 2: Chia train / val
        val_ratio_from_temp = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_from_temp,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )

        # Cập nhật stats
        self.stats['train_samples'] = len(X_train)
        self.stats['val_samples'] = len(X_val)
        self.stats['test_samples'] = len(X_test)

        print(f"\n   📈 KẾT QUẢ CHIA DỮ LIỆU:")
        print(f"   {'='*50}")
        print(f"   {'Set':<10} {'Samples':>12} {'Benign':>12} {'Attack':>12}")
        print(f"   {'-'*50}")
        print(f"   {'Train':<10} {len(X_train):>12,} {(y_train==0).sum():>12,} {(y_train==1).sum():>12,}")
        print(f"   {'Val':<10} {len(X_val):>12,} {(y_val==0).sum():>12,} {(y_val==1).sum():>12,}")
        print(f"   {'Test':<10} {len(X_test):>12,} {(y_test==0).sum():>12,} {(y_test==1).sum():>12,}")
        print(f"   {'-'*50}")
        print(f"   {'Total':<10} {len(X_train)+len(X_val)+len(X_test):>12,}")

        # Kiểm tra tỷ lệ
        print(f"\n   📊 TỶ LỆ ATTACK TRONG MỖI TẬP:")
        print(f"      Train: {(y_train==1).sum()/len(y_train)*100:.1f}%")
        print(f"      Val:   {(y_val==1).sum()/len(y_val)*100:.1f}%")
        print(f"      Test:  {(y_test==1).sum()/len(y_test)*100:.1f}%")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_training_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Lưu dữ liệu training

        Lưu các file:
        - X_train.npy, X_val.npy, X_test.npy
        - y_train.npy, y_val.npy, y_test.npy
        - scaler.pkl
        - training_metadata.json
        - feature_names.txt
        """
        print("\n" + "="*80)
        print("💾 ĐANG LƯU DỮ LIỆU TRAINING...")
        print("="*80)

        # Lưu numpy arrays
        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'X_val.npy', X_val)
        np.save(self.output_dir / 'X_test.npy', X_test)
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'y_val.npy', y_val)
        np.save(self.output_dir / 'y_test.npy', y_test)

        print(f"   ✅ X_train.npy: {X_train.shape}")
        print(f"   ✅ X_val.npy:   {X_val.shape}")
        print(f"   ✅ X_test.npy:  {X_test.shape}")
        print(f"   ✅ y_train.npy: {y_train.shape}")
        print(f"   ✅ y_val.npy:   {y_val.shape}")
        print(f"   ✅ y_test.npy:  {y_test.shape}")

        # Lưu scaler
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✅ scaler.pkl")

        # Lưu feature names
        with open(self.output_dir / 'feature_names.txt', 'w') as f:
            for name in self.feature_names:
                f.write(name + '\n')
        print(f"   ✅ feature_names.txt")

        # Chuẩn bị metadata
        metadata = {
            'n_features': len(self.feature_names),
            'input_shape': [int(X_train.shape[1]), int(X_train.shape[2])],
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'total_samples': int(X_train.shape[0] + X_val.shape[0] + X_test.shape[0]),
            'class_distribution': {
                'train': {
                    'benign': int((y_train == 0).sum()),
                    'attack': int((y_train == 1).sum())
                },
                'val': {
                    'benign': int((y_val == 0).sum()),
                    'attack': int((y_val == 1).sum())
                },
                'test': {
                    'benign': int((y_test == 0).sum()),
                    'attack': int((y_test == 1).sum())
                }
            },
            'benign_ratio': float(self.benign_ratio),
            'attack_ratio': float(self.attack_ratio),
            'preprocessing': {
                'log_transform': 'log_e(1+x)',
                'normalization': 'StandardScaler'
            },
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.output_dir / 'training_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"   ✅ training_metadata.json")

        print(f"\n📁 Tất cả file được lưu tại: {self.output_dir}")

    def calculate_class_weights(self, y_train):
        """
        Tính class weights cho training

        Sử dụng khi dữ liệu vẫn còn imbalanced
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))

        print(f"\n⚖️ CLASS WEIGHTS (cho training):")
        print(f"   Class 0 (Benign): {class_weights[0]:.4f}")
        print(f"   Class 1 (Attack): {class_weights[1]:.4f}")

        # Lưu class weights
        with open(self.output_dir / 'class_weights.pkl', 'wb') as f:
            pickle.dump(class_weights, f)
        print(f"   ✅ Đã lưu class_weights.pkl")

        return class_weights


def main():
    """Hàm chính"""

    print("\n" + "="*80)
    print("📊 BƯỚC 2: CHUẨN BỊ DỮ LIỆU TRAINING CHO CNN")
    print("   Cân bằng và chia train/val/test")
    print("="*80)

    print(f"\n📋 CẤU HÌNH:")
    print(f"   - Tổng mẫu mong muốn: {TOTAL_SAMPLES:,}")
    print(f"   - Tỷ lệ Benign: {BENIGN_RATIO*100:.0f}%")
    print(f"   - Tỷ lệ Attack: {ATTACK_RATIO*100:.0f}%")
    print(f"   - Train/Val/Test: {(1-TEST_SIZE-VAL_SIZE)*100:.0f}%/{VAL_SIZE*100:.0f}%/{TEST_SIZE*100:.0f}%")

    # Khởi tạo preparer
    preparer = TrainingDataPreparer(
        cleaned_data_dir=CLEANED_DATA_DIR,
        output_dir=OUTPUT_DIR,
        total_samples=TOTAL_SAMPLES,
        benign_ratio=BENIGN_RATIO,
        attack_ratio=ATTACK_RATIO,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE
    )

    # Bước 1: Đọc dữ liệu đã clean
    df = preparer.load_cleaned_data()

    # Bước 2: Cân bằng dữ liệu
    df = preparer.balanced_sample(df)

    # Tách features và labels
    X = df.drop(columns=['binary_label']).values
    y = df['binary_label'].values

    # Giải phóng bộ nhớ DataFrame
    del df
    gc.collect()

    # Bước 3: Áp dụng Log Transform
    X = preparer.apply_log_transform(X)

    # Bước 4: Chuẩn hóa
    X = preparer.normalize_features(X)

    # Bước 5: Reshape cho CNN
    X = preparer.reshape_for_cnn(X)

    # Bước 6: Chia train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = preparer.split_data(X, y)

    # Giải phóng bộ nhớ
    del X, y
    gc.collect()

    # Bước 7: Tính class weights
    class_weights = preparer.calculate_class_weights(y_train)

    # Bước 8: Lưu dữ liệu
    preparer.save_training_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\n" + "="*80)
    print("✅ HOÀN THÀNH BƯỚC 2!")
    print("   Dữ liệu đã sẵn sàng cho việc huấn luyện CNN.")
    print("   Chạy step3_train_cnn.py để train mô hình.")
    print("="*80)

    return preparer


if __name__ == "__main__":
    preparer = main()


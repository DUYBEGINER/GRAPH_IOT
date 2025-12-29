"""
BƯỚC 2: CHUẨN BỊ DỮ LIỆU TRAINING - CÂN BẰNG VÀ CHIA TRAIN/VAL/TEST
- Cân bằng nhãn theo tỷ lệ (70% Benign, 30% Attack)
- Log Transform: log_e(1+x) và StandardScaler
- Reshape cho CNN 1D
- Chia train 70%, val 10%, test 20%
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

IS_KAGGLE = os.path.exists('/kaggle/input')

# Cấu hình đường dẫn
if IS_KAGGLE:
    CLEANED_DATA_DIR = "/kaggle/working/cleaned_data"
    OUTPUT_DIR = "/kaggle/working/training_data"
    print("🌐 Kaggle")
else:
    CLEANED_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\cleaned_data"
    OUTPUT_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\training_data"
    print("💻 Local")

# Cấu hình cân bằng dữ liệu
RANDOM_STATE = 42
TOTAL_SAMPLES = 3000000
BENIGN_RATIO = 0.70
ATTACK_RATIO = 0.30
TARGET_BENIGN = int(TOTAL_SAMPLES * BENIGN_RATIO)
TARGET_ATTACK = int(TOTAL_SAMPLES * ATTACK_RATIO)

# Tỷ lệ chia
TEST_SIZE = 0.20
VAL_SIZE = 0.10

class TrainingDataPreparer:
    """Chuẩn bị dữ liệu training cho CNN: cân bằng, normalize, chia train/val/test"""

    def __init__(self, cleaned_data_dir, output_dir,
                 total_samples=TOTAL_SAMPLES,
                 benign_ratio=BENIGN_RATIO,
                 attack_ratio=ATTACK_RATIO,
                 test_size=TEST_SIZE,
                 val_size=VAL_SIZE):
        self.cleaned_data_dir = Path(cleaned_data_dir)
        self.output_dir = Path(output_dir)
        self.total_samples = total_samples
        self.benign_ratio = benign_ratio
        self.attack_ratio = attack_ratio
        self.test_size = test_size
        self.val_size = val_size

        self.target_benign = int(total_samples * benign_ratio)
        self.target_attack = int(total_samples * attack_ratio)

        self.scaler = StandardScaler()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = None

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
        print("\n" + "="*80)
        print(" ĐỌC DỮ LIỆU")
        print("="*80)

        parquet_path = self.cleaned_data_dir / 'cleaned_data.parquet'
        if not parquet_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {parquet_path}\nChạy step1_clean_data.py trước!")

        df = pd.read_parquet(parquet_path)

        feature_names_path = self.cleaned_data_dir / 'feature_names.txt'
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        else:
            self.feature_names = [col for col in df.columns if col != 'binary_label']

        self.stats['original_benign'] = int((df['binary_label'] == 0).sum())
        self.stats['original_attack'] = int((df['binary_label'] == 1).sum())
        self.stats['n_features'] = len(self.feature_names)

        print(f"   {len(df):,} mẫu - Benign: {self.stats['original_benign']:,}, Attack: {self.stats['original_attack']:,}")
        print(f"   Features: {self.stats['n_features']}")

        return df

    def balanced_sample(self, df):
        print("\n" + "="*80)
        print(" CÂN BẰNG DỮ LIỆU")
        print("="*80)

        df_benign = df[df['binary_label'] == 0]
        df_attack = df[df['binary_label'] == 1]

        n_benign = len(df_benign)
        n_attack = len(df_attack)

        print(f"   Target: {self.total_samples:,} (Benign {self.benign_ratio*100:.0f}%, Attack {self.attack_ratio*100:.0f}%)")

        actual_attack = min(self.target_attack, n_attack)
        actual_benign = int(actual_attack * (self.benign_ratio / self.attack_ratio))
        actual_benign = min(actual_benign, n_benign)

        if actual_benign < int(actual_attack * (self.benign_ratio / self.attack_ratio)):
            actual_attack = int(actual_benign * (self.attack_ratio / self.benign_ratio))

        print(f"   Lấy: Benign {actual_benign:,}, Attack {actual_attack:,} = {actual_benign + actual_attack:,}")
        print(f"   Tỷ lệ: {actual_benign/(actual_benign+actual_attack)*100:.1f}% - {actual_attack/(actual_benign+actual_attack)*100:.1f}%")

        df_benign_sampled = df_benign.sample(n=actual_benign, random_state=RANDOM_STATE)
        df_attack_sampled = df_attack.sample(n=actual_attack, random_state=RANDOM_STATE)

        df_balanced = pd.concat([df_benign_sampled, df_attack_sampled], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        self.stats['sampled_benign'] = actual_benign
        self.stats['sampled_attack'] = actual_attack

        del df_benign, df_attack, df_benign_sampled, df_attack_sampled
        gc.collect()

        return df_balanced

    def apply_log_transform(self, X):
        print("\n Log Transform: log_e(1+x)")

        min_val = X.min()
        if min_val < 0:
            X = X - min_val

        X_log = np.log1p(X)
        print(f"   Range: [{X.min():.4f}, {X.max():.4f}] -> [{X_log.min():.4f}, {X_log.max():.4f}]")

        return X_log

    def normalize_features(self, X):
        print("\n StandardScaler")
        X_normalized = self.scaler.fit_transform(X)
        print(f"   Mean: {X_normalized.mean():.6f}, Std: {X_normalized.std():.6f}")
        return X_normalized

    def reshape_for_cnn(self, X):
        print(f"\n Reshape: {X.shape} -> ", end="")
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        print(f"{X_reshaped.shape}")
        return X_reshaped

    def split_data(self, X, y):
        print("\n CHIA TRAIN/VAL/TEST")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=RANDOM_STATE, stratify=y
        )

        val_ratio_from_temp = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_from_temp, random_state=RANDOM_STATE, stratify=y_temp
        )

        self.stats['train_samples'] = len(X_train)
        self.stats['val_samples'] = len(X_val)
        self.stats['test_samples'] = len(X_test)

        print(f"\n   Train: {len(X_train):,} (Benign: {(y_train==0).sum():,}, Attack: {(y_train==1).sum():,})")
        print(f"   Val:   {len(X_val):,} (Benign: {(y_val==0).sum():,}, Attack: {(y_val==1).sum():,})")
        print(f"   Test:  {len(X_test):,} (Benign: {(y_test==0).sum():,}, Attack: {(y_test==1).sum():,})")
        print(f"   Tỷ lệ Attack: Train {(y_train==1).sum()/len(y_train)*100:.1f}%, Val {(y_val==1).sum()/len(y_val)*100:.1f}%, Test {(y_test==1).sum()/len(y_test)*100:.1f}%")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_training_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print("\n" + "="*80)
        print(" LƯU DỮ LIỆU")
        print("="*80)

        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'X_val.npy', X_val)
        np.save(self.output_dir / 'X_test.npy', X_test)
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'y_val.npy', y_val)
        np.save(self.output_dir / 'y_test.npy', y_test)

        print(f"   ✅ Đã lưu arrays: X_train{X_train.shape}, X_val{X_val.shape}, X_test{X_test.shape}")

        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(self.output_dir / 'feature_names.txt', 'w') as f:
            for name in self.feature_names:
                f.write(name + '\n')

        print(f"   ✅ scaler.pkl, feature_names.txt")

        metadata = {
            'n_features': len(self.feature_names),
            'input_shape': [int(X_train.shape[1]), int(X_train.shape[2])],
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'total_samples': int(X_train.shape[0] + X_val.shape[0] + X_test.shape[0]),
            'class_distribution': {
                'train': {'benign': int((y_train == 0).sum()), 'attack': int((y_train == 1).sum())},
                'val': {'benign': int((y_val == 0).sum()), 'attack': int((y_val == 1).sum())},
                'test': {'benign': int((y_test == 0).sum()), 'attack': int((y_test == 1).sum())}
            },
            'benign_ratio': float(self.benign_ratio),
            'attack_ratio': float(self.attack_ratio),
            'preprocessing': {'log_transform': 'log_e(1+x)', 'normalization': 'StandardScaler'},
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.output_dir / 'training_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)


    def calculate_class_weights(self, y_train):
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))

        print(f"\n⚖️ Class Weights: Benign {class_weights[0]:.4f}, Attack {class_weights[1]:.4f}")

        with open(self.output_dir / 'class_weights.pkl', 'wb') as f:
            pickle.dump(class_weights, f)

        return class_weights


def main():
    print("\n" + "="*80)
    print("📊 BƯỚC 2: CHUẨN BỊ DỮ LIỆU TRAINING")
    print("="*80)
    print(f"   Target: {TOTAL_SAMPLES:,} ({BENIGN_RATIO*100:.0f}% Benign, {ATTACK_RATIO*100:.0f}% Attack)")

    preparer = TrainingDataPreparer(
        cleaned_data_dir=CLEANED_DATA_DIR,
        output_dir=OUTPUT_DIR,
        total_samples=TOTAL_SAMPLES,
        benign_ratio=BENIGN_RATIO,
        attack_ratio=ATTACK_RATIO,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE
    )

    df = preparer.load_cleaned_data()
    df = preparer.balanced_sample(df)

    X = df.drop(columns=['binary_label']).values
    y = df['binary_label'].values
    del df
    gc.collect()

    X = preparer.apply_log_transform(X)
    X = preparer.normalize_features(X)
    X = preparer.reshape_for_cnn(X)

    X_train, X_val, X_test, y_train, y_val, y_test = preparer.split_data(X, y)
    del X, y
    gc.collect()

    class_weights = preparer.calculate_class_weights(y_train)
    preparer.save_training_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\n✅ HOÀN THÀNH! Chạy train_cnn.py để train model")

    return preparer


if __name__ == "__main__":
    preparer = main()


"""
Data Preprocessing Module for GNN Anomaly Detection
====================================================
Module này xử lý việc load, clean và chuẩn bị dữ liệu cho việc xây dựng graph.

Các bước xử lý:
1. Load dataset từ CSV
2. Xử lý missing values và infinite values
3. Chuyển đổi label sang binary (Normal/Anomaly)
4. Chuẩn hóa features
5. Lưu processed data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config


def load_data(file_path: str = None, sample_size: int = None) -> pd.DataFrame:
    """
    Load dataset từ CSV file.

    Args:
        file_path: Đường dẫn đến file CSV. Nếu None, sử dụng config.DATA_PATH
        sample_size: Số lượng samples để load. Nếu None, load toàn bộ

    Returns:
        DataFrame chứa dữ liệu
    """
    if file_path is None:
        file_path = config.DATA_PATH

    print(f"[INFO] Loading data from: {file_path}")

    # Load data với chỉ định low_memory=False để tránh warning
    if sample_size is not None:
        # Đọc header để lấy tên cột
        df_header = pd.read_csv(file_path, nrows=0)
        # Đọc toàn bộ và sample
        df = pd.read_csv(file_path, low_memory=False)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=config.RANDOM_SEED)
            print(f"[INFO] Sampled {sample_size} rows from {len(df)} total rows")
    else:
        df = pd.read_csv(file_path, low_memory=False)

    print(f"[INFO] Loaded data shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    return df


def explore_data(df: pd.DataFrame) -> dict:
    """
    Khám phá dữ liệu và trả về thông tin thống kê.

    Args:
        df: DataFrame cần phân tích

    Returns:
        Dictionary chứa các thông tin thống kê
    """
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'label_distribution': df[config.LABEL_COLUMN].value_counts().to_dict() if config.LABEL_COLUMN in df.columns else None
    }

    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    print(f"Shape: {stats['shape']}")
    print(f"\nMissing values per column:")
    for col, count in stats['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count}")

    if stats['label_distribution']:
        print(f"\nLabel distribution:")
        for label, count in stats['label_distribution'].items():
            print(f"  {label}: {count}")

    return stats


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu: xử lý missing values, infinite values, và drop columns không cần thiết.

    Args:
        df: DataFrame gốc

    Returns:
        DataFrame đã được làm sạch
    """
    print("\n[INFO] Cleaning data...")

    # Make a copy
    df = df.copy()

    # 1. Drop columns không cần thiết
    cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"[INFO] Dropped columns: {cols_to_drop}")

    # 2. Lấy danh sách feature columns (tất cả trừ Label)
    feature_cols = [col for col in df.columns if col != config.LABEL_COLUMN]

    # 3. Chuyển đổi kiểu dữ liệu sang numeric
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Xử lý infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # 5. Xử lý missing values - fill với median của cột
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    # 6. Drop rows với missing label
    if df[config.LABEL_COLUMN].isnull().sum() > 0:
        df = df.dropna(subset=[config.LABEL_COLUMN])

    print(f"[INFO] Cleaned data shape: {df.shape}")

    return df


def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển đổi labels thành binary: Normal (0) vs Anomaly (1).

    Args:
        df: DataFrame với cột Label

    Returns:
        DataFrame với cột binary_label mới
    """
    print("\n[INFO] Creating binary labels...")

    df = df.copy()

    # Tạo binary label: 0 = Normal (Benign), 1 = Anomaly (Attack)
    df['binary_label'] = (df[config.LABEL_COLUMN] != config.NORMAL_LABEL).astype(int)

    # Thống kê
    label_counts = df['binary_label'].value_counts()
    print(f"[INFO] Binary label distribution:")
    print(f"  Normal (0): {label_counts.get(0, 0)}")
    print(f"  Anomaly (1): {label_counts.get(1, 0)}")

    return df


def normalize_features(df: pd.DataFrame, scaler_type: str = None, fit: bool = True, scaler=None):
    """
    Chuẩn hóa các features.

    Args:
        df: DataFrame chứa features
        scaler_type: Loại scaler ('standard', 'minmax', 'robust')
        fit: True nếu cần fit scaler, False nếu dùng scaler đã có
        scaler: Scaler đã fit (khi fit=False)

    Returns:
        Tuple (normalized_features array, scaler, feature_names)
    """
    if scaler_type is None:
        scaler_type = config.SCALER_TYPE

    print(f"\n[INFO] Normalizing features using {scaler_type} scaler...")

    # Lấy feature columns (không bao gồm Label và binary_label)
    exclude_cols = [config.LABEL_COLUMN, 'binary_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Lấy feature values
    X = df[feature_cols].values.astype(np.float32)

    # Xử lý các giá trị NaN còn sót lại
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        # Khởi tạo scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        # Fit và transform
        X_normalized = scaler.fit_transform(X)
    else:
        # Chỉ transform
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        X_normalized = scaler.transform(X)

    print(f"[INFO] Normalized features shape: {X_normalized.shape}")

    return X_normalized, scaler, feature_cols


def preprocess_pipeline(file_path: str = None, sample_size: int = None, save_outputs: bool = True):
    """
    Pipeline hoàn chỉnh để preprocess data.

    Args:
        file_path: Đường dẫn đến file CSV
        sample_size: Số lượng samples (None = tất cả)
        save_outputs: Có lưu outputs hay không

    Returns:
        Dictionary chứa processed data và metadata
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)

    # 1. Load data
    if sample_size is None:
        sample_size = config.SAMPLE_SIZE
    df = load_data(file_path, sample_size)

    # 2. Explore data
    stats = explore_data(df)

    # 3. Clean data
    df = clean_data(df)

    # 4. Create binary labels
    df = create_binary_labels(df)

    # 5. Normalize features
    X, scaler, feature_names = normalize_features(df)

    # 6. Get labels
    y = df['binary_label'].values
    original_labels = df[config.LABEL_COLUMN].values

    # 7. Split data
    print("\n[INFO] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=y
    )

    val_ratio_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=config.RANDOM_SEED,
        stratify=y_temp
    )

    print(f"[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Val set: {X_val.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")

    # 8. Package results
    results = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'stats': stats
    }

    # 9. Save outputs
    if save_outputs:
        save_path = os.path.join(config.OUTPUT_DIR, 'preprocessed_data.pkl')
        joblib.dump(results, save_path)
        print(f"\n[INFO] Saved preprocessed data to: {save_path}")

        # Save scaler separately
        scaler_path = os.path.join(config.OUTPUT_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Saved scaler to: {scaler_path}")

        # Save feature names
        features_path = os.path.join(config.OUTPUT_DIR, 'feature_names.txt')
        with open(features_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"[INFO] Saved feature names to: {features_path}")

    print("\n[INFO] Preprocessing completed!")

    return results


def load_preprocessed_data(file_path: str = None):
    """
    Load dữ liệu đã được preprocess.

    Args:
        file_path: Đường dẫn đến file preprocessed_data.pkl

    Returns:
        Dictionary chứa processed data
    """
    if file_path is None:
        file_path = os.path.join(config.OUTPUT_DIR, 'preprocessed_data.pkl')

    print(f"[INFO] Loading preprocessed data from: {file_path}")
    data = joblib.load(file_path)
    print(f"[INFO] Loaded data with keys: {list(data.keys())}")

    return data


if __name__ == "__main__":
    # Run preprocessing pipeline
    results = preprocess_pipeline()

    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Number of features: {results['num_features']}")
    print(f"Training samples: {results['X_train'].shape[0]}")
    print(f"Validation samples: {results['X_val'].shape[0]}")
    print(f"Test samples: {results['X_test'].shape[0]}")


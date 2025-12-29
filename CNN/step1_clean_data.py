"""
BƯỚC 1: CLEAN DỮ LIỆU CICIDS2018 CHO CNN
- Loại bỏ cột không cần thiết, zero-variance columns
- Xử lý NaN/Inf, duplicate
- Chuyển nhãn sang binary (Benign=0, Attack=1)
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

# Cấu hình đường dẫn
if IS_KAGGLE:
    DATA_DIR = "/kaggle/input/cicids2018"
    OUTPUT_DIR = "/kaggle/working/cleaned_data"
    print("🌐 Kaggle")
else:
    DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
    OUTPUT_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\cleaned_data"
    print("💻 Local")

# Cấu hình xử lý
CHUNK_SIZE = 300000
RANDOM_STATE = 42

# Cột cần loại bỏ
COLUMNS_TO_DROP = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp']
LABEL_COLUMN = 'Label'


class CICIDS2018_DataCleaner:
    """Clean dữ liệu CICIDS2018: loại bỏ cột thừa, xử lý NaN/Inf, duplicate, chuyển binary label"""

    def __init__(self, data_dir, output_dir, chunk_size=CHUNK_SIZE):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_names = None
        self.zero_variance_cols = []
        self.column_modes = {}

        self.stats = {
            'total_rows_read': 0,
            'rows_after_cleaning': 0,
            'duplicates_removed': 0,
            'nan_replaced': 0,
            'inf_replaced': 0,
            'zero_variance_cols_removed': 0,
            'benign_count': 0,
            'attack_count': 0,
            'feature_count': 0,
            'processing_time': 0.0
        }

    def _get_csv_files(self):
        csv_files = list(self.data_dir.glob("*_TrafficForML_CICFlowMeter.csv"))
        if not csv_files:
            csv_files = list(self.data_dir.glob("*.csv"))
            csv_files = [f for f in csv_files if not f.name.endswith('.zip')]

        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy file CSV trong {self.data_dir}")

        print(f"\n Tìm thấy {len(csv_files)} file CSV")
        return sorted(csv_files)

    def _clean_column_names(self, df):
        df.columns = df.columns.str.strip()
        return df

    def _drop_identification_columns(self, df):
        columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        return df

    def _convert_to_numeric(self, df):
        feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _convert_to_binary_label(self, df):
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{LABEL_COLUMN}'")

        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
        df = df[df[LABEL_COLUMN] != 'label']
        df['binary_label'] = (df[LABEL_COLUMN] != 'benign').astype(int)
        df = df.drop(columns=[LABEL_COLUMN])

        return df

    def _first_pass_collect_info(self, csv_files):
        print("\n" + "="*80)
        print("THU THẬP THÔNG TIN VÀ TÍNH MODE")
        print("="*80)

        all_columns = None
        column_value_counts = {}
        column_min_max = {}

        for csv_file in csv_files:
            chunk_iterator = pd.read_csv(csv_file, chunksize=self.chunk_size,
                                        low_memory=False, encoding='utf-8')

            for chunk in chunk_iterator:
                chunk = self._clean_column_names(chunk)
                chunk = self._drop_identification_columns(chunk)
                chunk = self._convert_to_numeric(chunk)

                if all_columns is None:
                    all_columns = [col for col in chunk.columns if col != LABEL_COLUMN]
                    for col in all_columns:
                        column_value_counts[col] = {}
                        column_min_max[col] = {'min': np.inf, 'max': -np.inf}

                for col in all_columns:
                    if col in chunk.columns:
                        col_data = chunk[col].replace([np.inf, -np.inf], np.nan)
                        valid_data = col_data.dropna()

                        if len(valid_data) > 0:
                            col_min = valid_data.min()
                            col_max = valid_data.max()
                            column_min_max[col]['min'] = min(column_min_max[col]['min'], col_min)
                            column_min_max[col]['max'] = max(column_min_max[col]['max'], col_max)

                            vc = valid_data.value_counts().head(10).to_dict()
                            for val, count in vc.items():
                                if val not in column_value_counts[col]:
                                    column_value_counts[col][val] = 0
                                column_value_counts[col][val] += count

                gc.collect()

        for col in all_columns:
            if column_min_max[col]['min'] == column_min_max[col]['max']:
                self.zero_variance_cols.append(col)

        for col in all_columns:
            if col not in self.zero_variance_cols:
                if column_value_counts[col]:
                    mode_val = max(column_value_counts[col], key=column_value_counts[col].get)
                    self.column_modes[col] = mode_val
                else:
                    self.column_modes[col] = 0

        self.stats['zero_variance_cols_removed'] = len(self.zero_variance_cols)
        print(f"   Loại bỏ {len(self.zero_variance_cols)} cột zero-variance")
        print(f"   Giữ lại {len(all_columns) - len(self.zero_variance_cols)} features")

        return all_columns

    def _handle_nan_inf_with_mode(self, df):
        feature_cols = [col for col in df.columns if col != 'binary_label']

        for col in feature_cols:
            if col in self.column_modes:
                mode_val = self.column_modes[col]

                inf_mask = np.isinf(df[col])
                nan_mask = df[col].isna()

                self.stats['inf_replaced'] += inf_mask.sum()
                self.stats['nan_replaced'] += nan_mask.sum()

                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(mode_val)

        return df

    def _drop_zero_variance_columns(self, df):
        cols_to_drop = [col for col in self.zero_variance_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df

    def _process_single_file(self, csv_file):
        print(f"\n📄 {csv_file.name}")

        processed_chunks = []
        chunk_iterator = pd.read_csv(csv_file, chunksize=self.chunk_size,
                                     low_memory=False, encoding='utf-8')

        if TQDM_AVAILABLE:
            file_size = csv_file.stat().st_size
            estimated_chunks = max(1, file_size // (self.chunk_size * 500))
            chunk_iterator = tqdm(chunk_iterator, desc="   Chunks",
                                  total=estimated_chunks, unit="chunk")

        for chunk in chunk_iterator:
            self.stats['total_rows_read'] += len(chunk)

            chunk = self._clean_column_names(chunk)
            chunk = self._drop_identification_columns(chunk)
            chunk = self._convert_to_numeric(chunk)
            chunk = self._convert_to_binary_label(chunk)
            chunk = self._drop_zero_variance_columns(chunk)
            chunk = self._handle_nan_inf_with_mode(chunk)

            processed_chunks.append(chunk)
            gc.collect()

        if processed_chunks:
            df = pd.concat(processed_chunks, ignore_index=True)
            del processed_chunks
            gc.collect()
            print(f"   ✅ {len(df):,} mẫu")
            return df

        return None

    def clean_all_files(self):
        start_time = datetime.now()
        print("\n" + "="*80)
        print(" CLEAN DỮ LIỆU CICIDS2018")
        print("="*80)

        csv_files = self._get_csv_files()
        all_columns = self._first_pass_collect_info(csv_files)

        print("\n" + "="*80)
        print("🔧 CLEAN DỮ LIỆU")
        print("="*80)

        all_dataframes = []
        for csv_file in csv_files:
            df = self._process_single_file(csv_file)
            if df is not None:
                all_dataframes.append(df)

        print("\n   Gộp dữ liệu và loại duplicate...")
        df_combined = pd.concat(all_dataframes, ignore_index=True)
        del all_dataframes
        gc.collect()

        rows_before = len(df_combined)
        df_combined = df_combined.drop_duplicates()
        rows_after = len(df_combined)
        self.stats['duplicates_removed'] = rows_before - rows_after

        self.stats['benign_count'] = int((df_combined['binary_label'] == 0).sum())
        self.stats['attack_count'] = int((df_combined['binary_label'] == 1).sum())
        self.stats['rows_after_cleaning'] = len(df_combined)
        self.stats['feature_count'] = len(df_combined.columns) - 1

        self.feature_names = [col for col in df_combined.columns if col != 'binary_label']

        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()

        return df_combined

    def save_cleaned_data(self, df):
        print("\n" + "="*80)
        print(" LƯU DỮ LIỆU")
        print("="*80)

        parquet_path = self.output_dir / 'cleaned_data.parquet'
        df.to_parquet(parquet_path, index=False)
        print(f"   ✅ cleaned_data.parquet ({parquet_path.stat().st_size / (1024*1024):.1f} MB)")

        with open(self.output_dir / 'feature_names.txt', 'w') as f:
            for name in self.feature_names:
                f.write(name + '\n')

        with open(self.output_dir / 'column_modes.pkl', 'wb') as f:
            pickle.dump(self.column_modes, f)

        with open(self.output_dir / 'zero_variance_cols.pkl', 'wb') as f:
            pickle.dump(self.zero_variance_cols, f)

        stats_native = {}
        for key, value in self.stats.items():
            if hasattr(value, 'item'):
                stats_native[key] = value.item()
            elif isinstance(value, (np.integer, np.floating)):
                stats_native[key] = int(value) if isinstance(value, np.integer) else float(value)
            else:
                stats_native[key] = value

        metadata = {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'total_samples': int(len(df)),
            'benign_count': self.stats['benign_count'],
            'attack_count': self.stats['attack_count'],
            'benign_ratio': self.stats['benign_count'] / len(df),
            'attack_ratio': self.stats['attack_count'] / len(df),
            'zero_variance_cols': self.zero_variance_cols,
            'stats': stats_native,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.output_dir / 'cleaning_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        print(f"\n Đã lưu vào: {self.output_dir}")

    def print_summary(self):
        print("\n" + "="*80)
        print(" TÓM TẮT")
        print("="*80)
        print(f"   Tổng số dòng đọc:     {self.stats['total_rows_read']:,}")
        print(f"   Sau khi clean:        {self.stats['rows_after_cleaning']:,}")
        print(f"   Duplicate loại bỏ:    {self.stats['duplicates_removed']:,}")
        print(f"   NaN thay thế:         {self.stats['nan_replaced']:,}")
        print(f"   Inf thay thế:         {self.stats['inf_replaced']:,}")
        print(f"   Zero-variance loại:   {self.stats['zero_variance_cols_removed']}")
        print(f"   Features:             {self.stats['feature_count']}")
        print(f"\n   Benign:  {self.stats['benign_count']:,} ({self.stats['benign_count']/self.stats['rows_after_cleaning']*100:.1f}%)")
        print(f"   Attack:  {self.stats['attack_count']:,} ({self.stats['attack_count']/self.stats['rows_after_cleaning']*100:.1f}%)")
        print(f"\n   Thời gian: {self.stats['processing_time']:.2f}s")
        print("="*80)


def main():
    print("\n" + "="*80)
    print("🧹 BƯỚC 1: CLEAN DỮ LIỆU CICIDS2018 CHO CNN")
    print("="*80)

    cleaner = CICIDS2018_DataCleaner(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        chunk_size=CHUNK_SIZE
    )

    df = cleaner.clean_all_files()
    cleaner.save_cleaned_data(df)
    cleaner.print_summary()

    print("\n✅ HOÀN THÀNH!")
    print("   Chạy step2_prepare_training_data.py để chia train/val/test")

    return cleaner


if __name__ == "__main__":
    cleaner = main()


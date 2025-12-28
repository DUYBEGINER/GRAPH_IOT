"""
======================================================================================
BƯỚC 1: CLEAN VÀ TIỀN XỬ LÝ DATASET CICIDS2018 CHO MÔ HÌNH CNN
======================================================================================

Script này thực hiện các bước tiền xử lý dữ liệu:
1. Đọc từng file CSV theo chunks để tối ưu bộ nhớ
2. Loại bỏ các cột không cần thiết (IP, Port, Timestamp, Flow ID)
3. Loại bỏ các cột có variance = 0 (zero-variance columns)
4. Xử lý Infinity và NaN bằng Mode của cột
5. Loại bỏ các hàng trùng lặp
6. Chuyển đổi nhãn sang dạng binary (Benign=0, Attack=1)
7. Lưu dữ liệu đã clean vào folder để sử dụng sau

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
    OUTPUT_DIR = "/kaggle/working/cleaned_data"
    print("🌐 Đang chạy trên KAGGLE")
else:
    # Đường dẫn Local
    DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
    OUTPUT_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\cleaned_data"
    print("💻 Đang chạy trên LOCAL")

# ============================================================================
# CẤU HÌNH XỬ LÝ DỮ LIỆU
# ============================================================================

# Kích thước chunk khi đọc CSV (điều chỉnh theo RAM của máy)
CHUNK_SIZE = 300000  # 300k rows mỗi chunk

# Random state để tái tạo kết quả
RANDOM_STATE = 42

# ============================================================================
# DANH SÁCH CÁC CỘT CẦN LOẠI BỎ (Identification columns)
# ============================================================================

COLUMNS_TO_DROP = [
    'Flow ID',          # ID duy nhất cho mỗi flow - không có ý nghĩa phân loại
    'Src IP',           # IP nguồn - không tổng quát
    'Dst IP',           # IP đích - không tổng quát
    'Src Port',         # Port nguồn - có thể bị overfitting
    'Dst Port',         # Port đích - có thể bị overfitting
    'Timestamp',        # Thời gian - không liên quan đến pattern tấn công
]

# Cột nhãn
LABEL_COLUMN = 'Label'

# ============================================================================
# CLASS XỬ LÝ DỮ LIỆU
# ============================================================================

class CICIDS2018_DataCleaner:
    """
    Class clean dữ liệu CICIDS2018 cho mô hình CNN

    Các bước xử lý:
    1. Đọc dữ liệu theo chunks
    2. Loại bỏ cột identification
    3. Loại bỏ zero-variance columns
    4. Xử lý Infinity và NaN bằng Mode
    5. Loại bỏ duplicate
    6. Chuyển đổi nhãn sang binary
    7. Lưu dữ liệu đã clean
    """

    def __init__(self, data_dir, output_dir, chunk_size=CHUNK_SIZE):
        """
        Khởi tạo data cleaner

        Args:
            data_dir: Đường dẫn thư mục chứa file CSV
            output_dir: Đường dẫn thư mục lưu kết quả
            chunk_size: Số dòng mỗi chunk khi đọc CSV
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size

        # Tạo thư mục output nếu chưa có
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lưu tên các features và thông tin xử lý
        self.feature_names = None
        self.zero_variance_cols = []
        self.column_modes = {}  # Lưu mode của từng cột để xử lý NaN/Inf

        # Thống kê
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
            'processing_time': 0.0  # Float để lưu thời gian xử lý (giây)
        }

    def _get_csv_files(self):
        """Lấy danh sách các file CSV trong thư mục data"""
        csv_files = list(self.data_dir.glob("*_TrafficForML_CICFlowMeter.csv"))
        if not csv_files:
            # Thử pattern khác cho Kaggle
            csv_files = list(self.data_dir.glob("*.csv"))
            # Loại bỏ file zip nếu có
            csv_files = [f for f in csv_files if not f.name.endswith('.zip')]

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

    def _drop_identification_columns(self, df):
        """Loại bỏ các cột identification không cần thiết cho huấn luyện"""
        columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        return df

    def _convert_to_numeric(self, df):
        """Chuyển đổi các cột về dạng số"""
        feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
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

        # Xóa cột Label gốc, giữ lại binary_label
        df = df.drop(columns=[LABEL_COLUMN])

        return df

    def _first_pass_collect_info(self, csv_files):
        """
        Lần đọc đầu tiên: Thu thập thông tin về columns và tính mode

        Mục đích:
        - Xác định các cột có variance = 0
        - Tính mode của từng cột để thay thế NaN/Inf
        """
        print("\n" + "="*80)
        print("📊 BƯỚC 1: THU THẬP THÔNG TIN TỪ DỮ LIỆU")
        print("="*80)

        all_columns = None
        column_value_counts = {}  # Để tính mode
        column_min_max = {}  # Để kiểm tra variance

        for csv_file in csv_files:
            print(f"\n   Đang scan: {csv_file.name}")
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

                # Thu thập thông tin cho mỗi cột
                for col in all_columns:
                    if col in chunk.columns:
                        # Thay thế inf trước khi tính
                        col_data = chunk[col].replace([np.inf, -np.inf], np.nan)
                        valid_data = col_data.dropna()

                        if len(valid_data) > 0:
                            # Cập nhật min/max
                            col_min = valid_data.min()
                            col_max = valid_data.max()
                            column_min_max[col]['min'] = min(column_min_max[col]['min'], col_min)
                            column_min_max[col]['max'] = max(column_min_max[col]['max'], col_max)

                            # Thu thập value counts cho mode (lấy top 10 để tiết kiệm bộ nhớ)
                            vc = valid_data.value_counts().head(10).to_dict()
                            for val, count in vc.items():
                                if val not in column_value_counts[col]:
                                    column_value_counts[col][val] = 0
                                column_value_counts[col][val] += count

                gc.collect()

        # Xác định zero-variance columns
        print("\n   Đang phân tích variance của các cột...")
        for col in all_columns:
            if column_min_max[col]['min'] == column_min_max[col]['max']:
                self.zero_variance_cols.append(col)

        # Tính mode cho mỗi cột
        print("   Đang tính mode cho mỗi cột...")
        for col in all_columns:
            if col not in self.zero_variance_cols:
                if column_value_counts[col]:
                    # Mode là giá trị xuất hiện nhiều nhất
                    mode_val = max(column_value_counts[col], key=column_value_counts[col].get)
                    self.column_modes[col] = mode_val
                else:
                    self.column_modes[col] = 0  # Fallback nếu không có dữ liệu hợp lệ

        self.stats['zero_variance_cols_removed'] = len(self.zero_variance_cols)

        print(f"\n   ✅ Số cột zero-variance sẽ loại bỏ: {len(self.zero_variance_cols)}")
        if self.zero_variance_cols:
            print(f"      Các cột: {self.zero_variance_cols}")
        print(f"   ✅ Số cột sẽ giữ lại: {len(all_columns) - len(self.zero_variance_cols)}")

        return all_columns

    def _handle_nan_inf_with_mode(self, df):
        """
        Xử lý NaN và Infinity bằng Mode của cột

        Replace Infinity and NaN with the Mode of the column
        """
        feature_cols = [col for col in df.columns if col != 'binary_label']

        for col in feature_cols:
            if col in self.column_modes:
                mode_val = self.column_modes[col]

                # Đếm số lượng inf và nan
                inf_mask = np.isinf(df[col])
                nan_mask = df[col].isna()

                self.stats['inf_replaced'] += inf_mask.sum()
                self.stats['nan_replaced'] += nan_mask.sum()

                # Thay thế inf bằng nan trước
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

                # Thay thế tất cả nan bằng mode
                df[col] = df[col].fillna(mode_val)

        return df

    def _drop_zero_variance_columns(self, df):
        """Loại bỏ các cột có variance = 0"""
        cols_to_drop = [col for col in self.zero_variance_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
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
            file_size = csv_file.stat().st_size
            estimated_chunks = max(1, file_size // (self.chunk_size * 500))
            chunk_iterator = tqdm(chunk_iterator, desc="   Chunks",
                                  total=estimated_chunks, unit="chunk")

        for chunk in chunk_iterator:
            self.stats['total_rows_read'] += len(chunk)

            # Bước 1: Chuẩn hóa tên cột
            chunk = self._clean_column_names(chunk)

            # Bước 2: Loại bỏ cột identification
            chunk = self._drop_identification_columns(chunk)

            # Bước 3: Chuyển đổi sang dạng số
            chunk = self._convert_to_numeric(chunk)

            # Bước 4: Chuyển đổi nhãn sang binary
            chunk = self._convert_to_binary_label(chunk)

            # Bước 5: Loại bỏ zero-variance columns
            chunk = self._drop_zero_variance_columns(chunk)

            # Bước 6: Xử lý NaN và Inf bằng Mode
            chunk = self._handle_nan_inf_with_mode(chunk)

            processed_chunks.append(chunk)
            gc.collect()

        # Gộp các chunks lại
        if processed_chunks:
            df = pd.concat(processed_chunks, ignore_index=True)
            del processed_chunks
            gc.collect()
            return df

        return None

    def clean_all_files(self):
        """
        Clean tất cả các file CSV

        Returns:
            DataFrame đã clean hoàn chỉnh
        """
        start_time = datetime.now()
        print("\n" + "="*80)
        print(" BẮT ĐẦU CLEAN DỮ LIỆU CICIDS2018")
        print("="*80)

        csv_files = self._get_csv_files()

        # Bước 1: Thu thập thông tin (mode, zero-variance)
        all_columns = self._first_pass_collect_info(csv_files)

        # Bước 2: Xử lý từng file
        print("\n" + "="*80)
        print(" BƯỚC 2: CLEAN DỮ LIỆU")
        print("="*80)

        all_dataframes = []
        for csv_file in csv_files:
            df = self._process_single_file(csv_file)
            if df is not None:
                all_dataframes.append(df)
                print(f"   ✅ Đã xử lý: {len(df):,} mẫu")

        # Gộp tất cả lại
        print("\n" + "-"*80)
        print(" ĐANG GỘP DỮ LIỆU...")

        df_combined = pd.concat(all_dataframes, ignore_index=True)
        del all_dataframes
        gc.collect()

        print(f"   Tổng số mẫu sau khi gộp: {len(df_combined):,}")

        # Loại bỏ duplicate trên toàn bộ dataset
        print("   Đang loại bỏ duplicate...")
        rows_before = len(df_combined)
        df_combined = df_combined.drop_duplicates()
        rows_after = len(df_combined)
        self.stats['duplicates_removed'] = rows_before - rows_after
        print(f"   Số mẫu sau khi loại duplicate: {len(df_combined):,}")
        print(f"   Số duplicate đã loại: {self.stats['duplicates_removed']:,}")

        # Đếm số lượng mỗi class
        self.stats['benign_count'] = int((df_combined['binary_label'] == 0).sum())
        self.stats['attack_count'] = int((df_combined['binary_label'] == 1).sum())

        # Cập nhật thống kê
        self.stats['rows_after_cleaning'] = len(df_combined)
        self.stats['feature_count'] = len(df_combined.columns) - 1  # Trừ cột label

        # Lưu tên features
        self.feature_names = [col for col in df_combined.columns if col != 'binary_label']

        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()

        return df_combined

    def save_cleaned_data(self, df):
        """
        Lưu dữ liệu đã clean

        Lưu thành các file:
        - cleaned_data.parquet (dữ liệu đã clean, chưa normalize)
        - feature_names.txt
        - cleaning_metadata.json
        """
        print("\n" + "="*80)
        print(" ĐANG LƯU DỮ LIỆU ĐÃ CLEAN...")
        print("="*80)

        # Lưu dữ liệu dạng parquet (nhanh và nhỏ gọn)
        parquet_path = self.output_dir / 'cleaned_data.parquet'
        df.to_parquet(parquet_path, index=False)
        print(f"   ✅ Đã lưu: {parquet_path}")
        print(f"      Kích thước file: {parquet_path.stat().st_size / (1024*1024):.2f} MB")

        # Cũng lưu dạng CSV để dễ kiểm tra (optional, có thể comment nếu file quá lớn)
        # csv_path = self.output_dir / 'cleaned_data.csv'
        # df.to_csv(csv_path, index=False)
        # print(f"   ✅ Đã lưu: {csv_path}")

        # Lưu feature names
        with open(self.output_dir / 'feature_names.txt', 'w') as f:
            for name in self.feature_names:
                f.write(name + '\n')
        print(f"   ✅ Đã lưu: feature_names.txt")

        # Lưu column modes (để có thể sử dụng cho dữ liệu mới)
        with open(self.output_dir / 'column_modes.pkl', 'wb') as f:
            pickle.dump(self.column_modes, f)
        print(f"   ✅ Đã lưu: column_modes.pkl")

        # Lưu zero-variance columns
        with open(self.output_dir / 'zero_variance_cols.pkl', 'wb') as f:
            pickle.dump(self.zero_variance_cols, f)
        print(f"   ✅ Đã lưu: zero_variance_cols.pkl")

        # Chuyển đổi stats sang kiểu Python native
        stats_native = {}
        for key, value in self.stats.items():
            if hasattr(value, 'item'):
                stats_native[key] = value.item()
            elif isinstance(value, (np.integer, np.floating)):
                stats_native[key] = int(value) if isinstance(value, np.integer) else float(value)
            else:
                stats_native[key] = value

        # Lưu metadata
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
        print(f"   ✅ Đã lưu: cleaning_metadata.json")

        print(f"\n📁 Tất cả file được lưu tại: {self.output_dir}")

    def print_summary(self):
        """In tóm tắt quá trình xử lý"""
        print("\n" + "="*80)
        print(" TÓM TẮT CLEAN DỮ LIỆU")
        print("="*80)
        print(f"   Tổng số dòng đọc được:        {self.stats['total_rows_read']:,}")
        print(f"   Số dòng sau khi clean:        {self.stats['rows_after_cleaning']:,}")
        print(f"   Số duplicate đã loại:         {self.stats['duplicates_removed']:,}")
        print(f"   Số NaN đã thay bằng mode:     {self.stats['nan_replaced']:,}")
        print(f"   Số Inf đã thay bằng mode:     {self.stats['inf_replaced']:,}")
        print(f"   Số cột zero-variance đã loại: {self.stats['zero_variance_cols_removed']}")
        print(f"   Số features còn lại:          {self.stats['feature_count']}")
        print(f"\n   📈 PHÂN BỐ NHÃN:")
        print(f"   Số mẫu Benign (0):  {self.stats['benign_count']:,} ({self.stats['benign_count']/self.stats['rows_after_cleaning']*100:.1f}%)")
        print(f"   Số mẫu Attack (1):  {self.stats['attack_count']:,} ({self.stats['attack_count']/self.stats['rows_after_cleaning']*100:.1f}%)")
        print(f"\n   Thời gian xử lý: {self.stats['processing_time']:.2f} giây")
        print("="*80)


def main():
    """Hàm chính để chạy cleaning"""

    print("\n" + "="*80)
    print("🧹 BƯỚC 1: CLEAN DỮ LIỆU CICIDS2018 CHO CNN")
    print("   Phát hiện lưu lượng mạng IoT bất thường")
    print("="*80)

    # Khởi tạo cleaner
    cleaner = CICIDS2018_DataCleaner(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        chunk_size=CHUNK_SIZE
    )

    # Clean tất cả các file
    df = cleaner.clean_all_files()

    # Lưu dữ liệu đã clean
    cleaner.save_cleaned_data(df)

    # In tóm tắt
    cleaner.print_summary()

    print("\n✅ HOÀN THÀNH BƯỚC 1!")
    print("   Dữ liệu đã được clean và lưu vào folder.")
    print("   Chạy step2_prepare_training_data.py để chia train/val/test và cân bằng dữ liệu.")

    return cleaner


if __name__ == "__main__":
    cleaner = main()


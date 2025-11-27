"""
Script liệt kê các đặc trưng (features/columns) trong từng file dataset CICIDS2018
Sử dụng pandas để phân tích chi tiết
"""

import pandas as pd
import os
from pathlib import Path

# ================== CẤU HÌNH ==================
DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
OUTPUT_FILE = "features_summary.txt"

def analyze_csv_features(csv_file):
    """
    Phân tích chi tiết các features của một file CSV

    Args:
        csv_file: Đường dẫn đến file CSV

    Returns:
        dict: Thông tin về các features
    """
    try:
        # Đọc file CSV với low_memory=False để tránh warning
        df = pd.read_csv(csv_file, low_memory=False)

        # Lấy thông tin cơ bản
        info = {
            'filename': csv_file.name,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
        }

        return info, df

    except Exception as e:
        return {'error': str(e), 'filename': csv_file.name}, None


def list_features_in_datasets(data_dir, output_file):
    """
    Liệt kê và phân tích tất cả các features trong dataset CICIDS2018

    Args:
        data_dir: Đường dẫn đến thư mục chứa các file CSV
        output_file: Tên file để lưu kết quả
    """
    print("=" * 100)
    print("SCRIPT LIỆT KÊ CÁC ĐẶC TRƯNG TRONG DATASET CICIDS2018 (PANDAS VERSION)")
    print("=" * 100)

    # Tìm tất cả các file CSV
    csv_files = sorted(Path(data_dir).glob("*_TrafficForML_CICFlowMeter.csv"))

    if not csv_files:
        print(f"⚠️  Không tìm thấy file CSV nào trong thư mục: {data_dir}")
        return

    print(f"\n📁 Thư mục dữ liệu: {data_dir}")
    print(f"📊 Tìm thấy {len(csv_files)} file CSV\n")

    # Lưu thông tin tất cả các file
    all_features_info = {}
    all_features_set = set()
    all_dtypes = {}

    # Mở file output
    output_path = os.path.join(data_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("DANH SÁCH CÁC ĐẶC TRƯNG (FEATURES) TRONG DATASET CICIDS2018\n")
        f.write("Phân tích chi tiết với pandas\n")
        f.write("=" * 100 + "\n\n")

        # Phân tích từng file
        for idx, csv_file in enumerate(csv_files, 1):
            filename = csv_file.name
            print(f"[{idx}/{len(csv_files)}] Đang xử lý: {filename}")

            info, df = analyze_csv_features(csv_file)

            if 'error' in info:
                print(f"    ✗ Lỗi: {info['error']}")
                f.write(f"\nFile: {filename}\n")
                f.write(f"LỖI: {info['error']}\n")
                continue

            # Lưu thông tin
            all_features_info[filename] = info
            all_features_set.update(info['columns'])

            # Lưu dtype của mỗi feature
            for col, dtype in info['dtypes'].items():
                if col not in all_dtypes:
                    all_dtypes[col] = {}
                all_dtypes[col][filename] = str(dtype)

            # Ghi vào file output
            f.write(f"\n{'=' * 100}\n")
            f.write(f"File: {filename}\n")
            f.write(f"{'=' * 100}\n")
            f.write(f"Số dòng:        {info['num_rows']:,}\n")
            f.write(f"Số cột:         {info['num_columns']}\n")
            f.write(f"Dung lượng RAM: {info['memory_usage']:.2f} MB\n")
            f.write(f"\n{'-' * 100}\n")
            f.write(f"{'STT':<5} {'TÊN CỘT':<50} {'KIỂU DỮ LIỆU':<20} {'MISSING VALUES':<15}\n")
            f.write(f"{'-' * 100}\n")

            for i, col in enumerate(info['columns'], 1):
                dtype = str(info['dtypes'][col])
                missing = info['missing_values'][col]
                f.write(f"{i:<5} {col:<50} {dtype:<20} {missing:<15}\n")

            # In ra màn hình
            print(f"    ✓ Số dòng: {info['num_rows']:,}")
            print(f"    ✓ Số cột: {info['num_columns']}")
            print(f"    ✓ RAM: {info['memory_usage']:.2f} MB")

            # Kiểm tra các cột có missing values
            missing_cols = [col for col, count in info['missing_values'].items() if count > 0]
            if missing_cols:
                print(f"    ⚠️  {len(missing_cols)} cột có missing values")

        # ============== TÓM TẮT TỔNG QUAN ==============
        print("\n" + "=" * 100)
        print("TÓM TẮT TỔNG QUAN")
        print("=" * 100)

        f.write(f"\n\n{'=' * 100}\n")
        f.write("TÓM TẮT TỔNG QUAN\n")
        f.write(f"{'=' * 100}\n\n")

        # Thống kê cơ bản
        total_files = len(all_features_info)
        total_features = len(all_features_set)
        total_rows = sum(info['num_rows'] for info in all_features_info.values() if 'num_rows' in info)
        total_memory = sum(info['memory_usage'] for info in all_features_info.values() if 'memory_usage' in info)

        summary_stats = [
            f"Tổng số file:              {total_files}",
            f"Tổng số features duy nhất: {total_features}",
            f"Tổng số dòng dữ liệu:      {total_rows:,}",
            f"Tổng dung lượng RAM:       {total_memory:.2f} MB",
        ]

        for stat in summary_stats:
            print(stat)
            f.write(stat + "\n")

        # Kiểm tra schema consistency
        print("\n" + "-" * 100)
        f.write("\n" + "-" * 100 + "\n")

        if all_features_info:
            first_file = list(all_features_info.keys())[0]
            first_columns = set(all_features_info[first_file]['columns'])
            all_same = True

            for filename, info in all_features_info.items():
                if set(info['columns']) != first_columns:
                    all_same = False
                    break

            if all_same:
                msg = "✓ Tất cả các file có cùng schema (các cột giống nhau và cùng thứ tự)"
                print(msg)
                f.write(msg + "\n")
            else:
                msg = "⚠️  Các file có schema khác nhau"
                print(msg)
                f.write(msg + "\n\n")
                f.write("CHI TIẾT SỰ KHÁC BIỆT:\n")
                f.write("-" * 100 + "\n")

                for filename, info in all_features_info.items():
                    file_columns = set(info['columns'])
                    missing = first_columns - file_columns
                    extra = file_columns - first_columns

                    if missing or extra:
                        f.write(f"\nFile: {filename}\n")
                        if missing:
                            f.write(f"  Thiếu cột: {sorted(missing)}\n")
                        if extra:
                            f.write(f"  Cột thêm: {sorted(extra)}\n")

        # ============== DANH SÁCH TẤT CẢ FEATURES ==============
        f.write(f"\n\n{'=' * 100}\n")
        f.write("DANH SÁCH TẤT CẢ CÁC FEATURES DUY NHẤT\n")
        f.write(f"{'=' * 100}\n\n")
        f.write(f"{'STT':<5} {'TÊN FEATURE':<60} {'KIỂU DỮ LIỆU PHỔ BIẾN':<30}\n")
        f.write("-" * 100 + "\n")

        for i, feature in enumerate(sorted(all_features_set), 1):
            # Xác định kiểu dữ liệu phổ biến nhất cho feature này
            if feature in all_dtypes:
                dtype_counts = {}
                for dtype in all_dtypes[feature].values():
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                most_common_dtype = max(dtype_counts, key=dtype_counts.get)
            else:
                most_common_dtype = "Unknown"

            f.write(f"{i:<5} {feature:<60} {most_common_dtype:<30}\n")

        # ============== PHÂN TÍCH KIỂU DỮ LIỆU ==============
        f.write(f"\n\n{'=' * 100}\n")
        f.write("PHÂN TÍCH KIỂU DỮ LIỆU CỦA CÁC FEATURES\n")
        f.write(f"{'=' * 100}\n\n")

        # Đếm các loại dtype
        dtype_summary = {}
        for feature, dtypes_dict in all_dtypes.items():
            for dtype in dtypes_dict.values():
                # Chuẩn hóa dtype name
                if 'int' in dtype:
                    dtype_category = 'Integer'
                elif 'float' in dtype:
                    dtype_category = 'Float'
                elif 'object' in dtype:
                    dtype_category = 'Object/String'
                else:
                    dtype_category = dtype

                dtype_summary[dtype_category] = dtype_summary.get(dtype_category, 0) + 1

        f.write("Phân bố kiểu dữ liệu:\n")
        f.write("-" * 100 + "\n")
        for dtype, count in sorted(dtype_summary.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {dtype:<30} {count:>5} features\n")

    print(f"\n✅ Kết quả đã được lưu vào: {output_path}")
    print("=" * 100)

    # Hiển thị preview các features
    if all_features_set:
        print("\n📋 DANH SÁCH 20 FEATURES ĐẦU TIÊN:")
        print("-" * 100)
        for i, feature in enumerate(sorted(all_features_set)[:20], 1):
            # Lấy dtype phổ biến
            if feature in all_dtypes:
                dtype_counts = {}
                for dtype in all_dtypes[feature].values():
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                most_common_dtype = max(dtype_counts, key=dtype_counts.get)
            else:
                most_common_dtype = "Unknown"

            print(f"  {i:2d}. {feature:<50} [{most_common_dtype}]")

        if len(all_features_set) > 20:
            print(f"\n  ... và {len(all_features_set) - 20} features khác")

        print("\n📊 PHÂN BỐ KIỂU DỮ LIỆU:")
        print("-" * 100)
        for dtype, count in sorted(dtype_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  {dtype:<30} {count:>5} features")

    return all_features_info, all_features_set


def main():
    """Hàm main để chạy script"""
    print("\n🚀 Bắt đầu phân tích dataset CICIDS2018...\n")

    features_info, all_features = list_features_in_datasets(DATA_DIR, OUTPUT_FILE)

    if features_info:
        print("\n" + "=" * 100)
        print("CHI TIẾT SỐ LƯỢNG DÒNG VÀ CỘT TRONG TỪNG FILE")
        print("=" * 100)
        print(f"{'FILE NAME':<60} {'DÒNG':>15} {'CỘT':>10} {'RAM (MB)':>12}")
        print("-" * 100)

        for filename, info in features_info.items():
            if 'num_rows' in info:
                print(f"{filename:<60} {info['num_rows']:>15,} {info['num_columns']:>10} {info['memory_usage']:>12.2f}")

        print("=" * 100)

    print("\n✅ Hoàn thành! Kiểm tra file features_summary.txt để xem chi tiết đầy đủ.")


if __name__ == "__main__":
    main()


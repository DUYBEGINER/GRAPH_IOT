"""
Script để trích xuất các nhãn (Label) từ các tệp CSV của CICIDS2018
"""

import pandas as pd
import os
from pathlib import Path

def extract_labels_from_csv(csv_file_path):
    """
    Trích xuất các nhãn từ một tệp CSV cụ thể
    
    Args:
        csv_file_path: Đường dẫn đến tệp CSV
    
    Returns:
        Set chứa các nhãn duy nhất trong tệp
    """
    try:
        # Đọc tệp CSV
        df = pd.read_csv(csv_file_path)
        
        # Lấy các nhãn duy nhất từ cột 'Label'
        if 'Label' in df.columns:
            labels = set(df['Label'].unique())
            return labels
        else:
            print(f"Cảnh báo: Không tìm thấy cột 'Label' trong {csv_file_path}")
            return set()
    except Exception as e:
        print(f"Lỗi khi đọc {csv_file_path}: {e}")
        return set()

def extract_labels_from_directory(directory_path):
    """
    Trích xuất tất cả các nhãn từ tất cả các tệp CSV trong thư mục
    
    Args:
        directory_path: Đường dẫn đến thư mục chứa các tệp CSV
    
    Returns:
        Dictionary với key là tên tệp và value là set các nhãn
    """
    all_labels_by_file = {}
    all_labels_combined = set()
    
    # Lấy tất cả các tệp CSV trong thư mục
    csv_files = list(Path(directory_path).glob("*.csv"))
    
    if not csv_files:
        print(f"Không tìm thấy tệp CSV nào trong {directory_path}")
        return all_labels_by_file, all_labels_combined
    
    print(f"Tìm thấy {len(csv_files)} tệp CSV")
    print("-" * 80)
    
    for csv_file in csv_files:
        print(f"\nĐang xử lý: {csv_file.name}")
        labels = extract_labels_from_csv(csv_file)
        all_labels_by_file[csv_file.name] = labels
        all_labels_combined.update(labels)
        
        if labels:
            print(f"Các nhãn tìm thấy ({len(labels)}): {sorted(labels)}")
    
    return all_labels_by_file, all_labels_combined

def main():
    """Hàm chính để chạy script"""
    
    # Đường dẫn đến thư mục chứa các tệp CSV
    csv_directory = r"D:\PROJECT\Machine Learning\IOT\CICIDS2018-CSV"
    
    print("=" * 80)
    print("TRÍCH XUẤT NHÃN TỪ CÁC TẾP CSV CICIDS2018")
    print("=" * 80)
    
    # Trích xuất nhãn từ tất cả các tệp
    labels_by_file, all_labels = extract_labels_from_directory(csv_directory)
    
    # Hiển thị kết quả tổng hợp
    print("\n" + "=" * 80)
    print("KẾT QUẢ TỔNG HỢP")
    print("=" * 80)
    print(f"\nTổng số nhãn duy nhất tìm thấy: {len(all_labels)}")
    print(f"Danh sách các nhãn: {sorted(all_labels)}")
    
    # Hiển thị số lượng nhãn trong mỗi tệp
    print("\n" + "-" * 80)
    print("SỐ LƯỢNG NHÃN TRONG MỖI TẾP")
    print("-" * 80)
    for filename, labels in sorted(labels_by_file.items()):
        print(f"{filename}: {len(labels)} nhãn")
    
    # Lưu kết quả ra tệp văn bản
    output_file = "labels_summary.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TỔNG HỢP CÁC NHÃN TỪ CICIDS2018 DATASET\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tổng số nhãn duy nhất: {len(all_labels)}\n\n")
        f.write("Danh sách các nhãn:\n")
        for label in sorted(all_labels):
            f.write(f"  - {label}\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("Chi tiết theo từng tệp:\n")
        f.write("-" * 80 + "\n\n")
        for filename, labels in sorted(labels_by_file.items()):
            f.write(f"{filename}:\n")
            f.write(f"  Số lượng: {len(labels)} nhãn\n")
            f.write(f"  Danh sách: {sorted(labels)}\n\n")
    
    print(f"\nKết quả đã được lưu vào tệp: {output_file}")

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

# ============================================================================
# CONFIGURATION (AUTO-DETECT KAGGLE VS LOCAL)
# ============================================================================
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    print("🌍 ENVIRONMENT: KAGGLE DETECTED")
    # Tự động tìm dataset CICIDS2018 trong input
    # Ưu tiên các folder thông dụng
    possible_dirs = [
        "/kaggle/input/cicids2018/CICIDS2018_CSV", 
        "/kaggle/input/cse-cic-ids2018",
        "/kaggle/input/cicids2018"
    ]
    RAW_DATA_DIR = None
    for d in possible_dirs:
        if os.path.exists(d):
            RAW_DATA_DIR = d
            break
    
    # Fallback nếu không tìm thấy đúng tên, lấy folder đầu tiên trong input
    if RAW_DATA_DIR is None:
        try:
            subdirs = glob.glob("/kaggle/input/*")
            if subdirs:
                RAW_DATA_DIR = subdirs[0]
        except:
            pass
            
    if RAW_DATA_DIR is None:
        RAW_DATA_DIR = "/kaggle/input" # Hy vọng user mount đúng
        
    OUTPUT_DIR = "/kaggle/working/processed_lstm"
    TARGET_ROWS = 2000000 # Kaggle RAM mạnh, lấy 2 triệu dòng
    print(f"   - Raw Data: {RAW_DATA_DIR}")
    print(f"   - Output:   {OUTPUT_DIR}")
    print(f"   - Target:   {TARGET_ROWS:,} rows")

else:
    print("💻 ENVIRONMENT: LOCAL DESKTOP DETECTED")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Local: Data nằm ở thư mục cha ../data_IOT
    RAW_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data_IOT")
    OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_lstm')
    TARGET_ROWS = 2000000  # Local RAM hạn chế, lấy 500k dòng
    print(f"   - Raw Data: {RAW_DATA_DIR}")
    print(f"   - Output:   {OUTPUT_DIR}")
    print(f"   - Target:   {TARGET_ROWS:,} rows")

# Common Params
WINDOW_SIZE = 10
BALANCE_DATA = True
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

def load_and_clean_raw_data():
    """Đọc trực tiếp từ CSV và làm sạch sơ bộ"""
    print(f"Checking data dir: {RAW_DATA_DIR}")
    
    # Tìm tất cả các file có đuôi .csv
    all_paths = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.csv"), recursive=True)
    csv_files = []
    
    for p in all_paths:
        if os.path.isfile(p):
            csv_files.append(p)
    
    if not csv_files:
        print("❌ Không tìm thấy file CSV thực sự!")
        exit(1)
        
    print(f"Found {len(csv_files)} actual CSV files.")
    
    dfs = []
    total_rows = 0
    
    # Sắp xếp để đảm bảo tính thứ tự thời gian (quan trọng cho LSTM)
    csv_files = sorted(csv_files)
    
    pbar = tqdm(csv_files, desc="Loading CSVs")
    for f in pbar:
        if total_rows >= TARGET_ROWS:
            break
            
        try:
            # Đọc file thực sự
            df = pd.read_csv(f, low_memory=False, nrows=200000) 
            
            # Clean Headers lặp lại (quan trọng vì dataset CICIDS2018 hay bị lỗi này)
            if 'Label' in df.columns:
                # Loại bỏ dòng mà cột Label chứa chữ 'Label' (header bị lặp)
                df = df[df['Label'].astype(str).str.lower() != 'label']
            
            # 2. Drop columns không dùng cho LSTM
            cols_to_drop = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port']
            cols_exist = [c for c in cols_to_drop if c in df.columns]
            df.drop(columns=cols_exist, inplace=True)
            
            # 3. Convert Numeric & Optimize Memory
            if 'Label' in df.columns:
                label_col = df['Label']
                df = df.drop(columns=['Label'])
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.astype(np.float32)
                df['Label'] = label_col
            else:
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.astype(np.float32)
            
            # 4. Fill NaNs
            df = df.fillna(0)
            
            if len(df) > 0:
                dfs.append(df)
                total_rows += len(df)
                pbar.set_postfix({'rows': f"{total_rows:,}"})
            
        except Exception as e:
            # print(f"Error reading {f}: {e}")
            pass

    print("\nMerging data...")
    if not dfs:
        print("❌ Không load được dữ liệu nào. Hãy kiểm tra lại file CSV bên trong data_IOT.")
        exit(1)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Handle Infinity
    print("Handling Infinity values...")
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns
    full_df[numeric_cols] = full_df[numeric_cols].replace([np.inf, -np.inf], 0)
    
    del dfs
    gc.collect()
    
    return full_df

def create_sequences_and_balance(df):
    """
    Tạo sequences và cân bằng dữ liệu Ở CẤP ĐỘ SEQUENCE.
    Điều này giữ nguyên tính toàn vẹn của sliding window.
    """
    # 1. Prepare Features & Label
    labels = df['Label'].values
    # Encode Label: Benign=0, Attack=1
    y_binary = (labels != 'Benign').astype(int)
    
    # Drop label col for features
    X_data = df.drop(columns=['Label']).values
    feature_names = df.drop(columns=['Label']).columns.tolist()
    
    # 2. Normalize (StandardScaler)
    print("Normalizing features...")
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data).astype(np.float32) # Keep float32
    
    # SAVE SCALER & FEATURE NAMES for Inference
    print("Saving scaler and feature names...")
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, 'feature_names.json'), 'w') as f:
        import json
        json.dump(feature_names, f)
    
    print(f"\nCreating Sliding Windows (Size: {WINDOW_SIZE})...")
    # Sử dụng logic sliding window
    # X_seq: (N, Window, Features)
    # y_seq: (N,) - Label của dòng cuối cùng trong window
    
    Xs, ys = [], []
    
    # Để tránh loop Python chậm, ta dùng loop đơn giản nhưng hiệu quả
    # Nếu dataset quá lớn, có thể dùng stride_tricks (nhưng cẩn thận RAM copy)
    
    # Chỉ lấy sequence hợp lệ (không bị ngắt quãng giữa các file nếu ghép)
    # Ở đây ta chấp nhận rủi ro nhỏ ở điểm nối các file để code đơn giản
    
    # Optimization: Chỉ loop qua index
    total_len = len(X_data) - WINDOW_SIZE
    
    # Cảnh báo nếu dữ liệu quá lớn
    if total_len > 1000000:
        print("Data lớn, quá trình tạo sequence có thể mất vài phút...")

    # Tạo sequences
    # Cách nhanh hơn: List comprehension
    indices = np.arange(total_len)
    
    # Để tiết kiệm RAM, ta không tạo ngay numpy array khổng lồ
    # Ta sẽ tạo index cho Attack và Benign trước
    
    # Xác định label cho mỗi window (là label của dòng cuối cùng)
    # y_binary[WINDOW_SIZE:] khớp với index của window
    window_labels = y_binary[WINDOW_SIZE:]
    
    attack_indices = np.where(window_labels == 1)[0]
    benign_indices = np.where(window_labels == 0)[0]
    
    print(f"Found {len(attack_indices):,} attack sequences")
    print(f"Found {len(benign_indices):,} benign sequences")
    
    if BALANCE_DATA:
        print("\nBalancing Data (Undersampling Benign Sequences)...")
        # Lấy tất cả Attack
        # Lấy Benign bằng số lượng Attack (hoặc tối đa nếu ít hơn)
        n_samples = min(len(attack_indices), len(benign_indices))
        
        if n_samples == 0:
            print("⚠️ Warning: Không tìm thấy Attack nào! Sẽ dùng toàn bộ data.")
            final_indices = indices
        else:
            # Random chọn Benign sequences
            np.random.seed(RANDOM_STATE)
            chosen_benign = np.random.choice(benign_indices, n_samples, replace=False)
            
            # Gộp lại
            final_indices = np.concatenate([attack_indices, chosen_benign])
            # Shuffle sequence order (Quan trọng: Shuffle CÁC SEQUENCE, không phải shuffle bên trong sequence)
            np.random.shuffle(final_indices)
            
            print(f"✓ Balanced Total Sequences: {len(final_indices):,} (50% Attack / 50% Benign)")
    else:
        final_indices = indices
        
    # Bây giờ mới build mảng X thật sự dựa trên index đã lọc
    # Điều này tiết kiệm cực nhiều RAM so với tạo hết rồi mới lọc
    print("Building final sequence array...")
    
    X_final = []
    y_final = []
    
    for idx in tqdm(final_indices, desc="Constructing 3D Array"):
        # idx là vị trí bắt đầu của window (trong logic cũ là 0..len-window)
        # window: X[idx : idx + WINDOW]
        # label: y[idx + WINDOW] -> chính là window_labels[idx]
        
        # Sửa lại logic index một chút cho khớp window_labels
        # window_labels[i] ứng với window kết thúc tại i+WINDOW
        # start index = i
        
        X_final.append(X_data[idx : idx + WINDOW_SIZE])
        y_final.append(window_labels[idx])
        
    return np.array(X_final, dtype=np.float32), np.array(y_final, dtype=np.float32)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("🚀 STARTED LSTM INDEPENDENT PREPROCESSING")
    
    # 1. Load Raw
    df = load_and_clean_raw_data()
    print(f"Raw Data Loaded: {df.shape}")
    
    # 2. Create Sequences & Balance
    X_seq, y_seq = create_sequences_and_balance(df)
    
    print(f"\nFinal Shape:")
    print(f"X: {X_seq.shape}")
    print(f"y: {y_seq.shape}")
    
    # 3. Split
    print("Splitting Train/Val/Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=TEST_SIZE, shuffle=False # Giữ thứ tự thời gian của các sequence đã shuffle (hoặc không)
    )
    # Lưu ý: Ở bước Balance trên ta đã shuffle thứ tự các sequence. 
    # Với LSTM Anomaly detection, thường ta split theo thời gian TRƯỚC khi balance.
    # Nhưng vì đây là bài toán Classification (Attack vs Benign) dựa trên window,
    # việc shuffle các window là chấp nhận được.
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, shuffle=False
    )
    
    # 4. Save
    print(f"Saving to {OUTPUT_DIR}...")
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    
    print("\n✅ DONE! Ready for training.")

if __name__ == "__main__":
    main()
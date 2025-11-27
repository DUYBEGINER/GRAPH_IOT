"""
Quick Start Script - Demo nhanh với sample data
Chạy script này để test pipeline với dữ liệu nhỏ
"""

import os
import sys

# ============================================================================
# QUICK START CONFIGURATION
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           GNN-BASED IOT ANOMALY DETECTION - QUICK START                   ║
║                                                                            ║
║  Script này sẽ chạy toàn bộ pipeline với sample nhỏ để test nhanh        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

print("Cấu hình Quick Start:")
print("-" * 80)
print("  ✓ Sample size: 10,000 rows (thay vì toàn bộ dataset)")
print("  ✓ K-neighbors: 5 (thay vì 10)")
print("  ✓ Hidden channels: 64 (thay vì 128)")
print("  ✓ Num epochs: 50 (thay vì 100)")
print("  ✓ Model: GCN (nhanh nhất)")
print("-" * 80)
print()

response = input("Bạn có muốn tiếp tục? (y/n): ")

if response.lower() != 'y':
    print("Đã hủy.")
    sys.exit(0)

# ============================================================================
# UPDATE CONFIGURATION FILES
# ============================================================================

print("\nĐang cấu hình pipeline cho quick start...")

# Update preprocess_data.py
preprocess_config = """
DATA_DIR = r"D:\\PROJECT\\Machine Learning\\IOT\\CICIDS2018-CSV"
OUTPUT_DIR = r"D:\\PROJECT\\Machine Learning\\IOT\\processed_data"
SAMPLE_SIZE = 10000  # Quick start: chỉ dùng 10k samples
RANDOM_STATE = 42
"""

# Update build_graph.py
graph_config = """
K_NEIGHBORS = 5  # Quick start: giảm k-neighbors
SIMILARITY_THRESHOLD = 0.5
GRAPH_TYPE = 'knn'
"""

# Update train_gnn.py
train_config = """
MODEL_NAME = 'GCN'  # Quick start: dùng GCN (nhanh nhất)
HIDDEN_CHANNELS = 64  # Quick start: giảm model size
NUM_LAYERS = 2  # Quick start: giảm layers
DROPOUT = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS = 50  # Quick start: giảm epochs
PATIENCE = 10
TASK = 'binary'
"""

print("✓ Configuration updated")

# ============================================================================
# RUN PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("BẮT ĐẦU QUICK START PIPELINE")
print("=" * 80)
print()

try:
    # Step 1: Preprocessing
    print("BƯỚC 1/3: Preprocessing data...")
    print("-" * 80)

    # Temporarily modify preprocess_data.py
    import preprocess_data
    preprocess_data.SAMPLE_SIZE = 10000

    data, X, y_binary, y_multi, feature_cols = preprocess_data.main()
    print("✓ Preprocessing hoàn thành\n")

    # Step 2: Build graph
    print("BƯỚC 2/3: Building graph...")
    print("-" * 80)

    import build_graph
    build_graph.K_NEIGHBORS = 5
    build_graph.MAX_SAMPLES = 10000  # Limit samples for quick start

    graph_binary, graph_multi, metadata = build_graph.build_graph_dataset()
    print("✓ Graph construction hoàn thành\n")

    # Step 3: Train
    print("BƯỚC 3/3: Training GNN model...")
    print("-" * 80)

    import train_gnn
    train_gnn.MODEL_NAME = 'GCN'
    train_gnn.HIDDEN_CHANNELS = 64
    train_gnn.NUM_LAYERS = 2
    train_gnn.NUM_EPOCHS = 50
    train_gnn.PATIENCE = 10

    train_gnn.main()
    print("✓ Training hoàn thành\n")

    # Success!
    print("\n" + "=" * 80)
    print("🎉 QUICK START HOÀN THÀNH THÀNH CÔNG! 🎉")
    print("=" * 80)
    print()
    print("Kết quả đã được lưu tại:")
    print("  📁 Processed data: D:\\PROJECT\\Machine Learning\\IOT\\processed_data")
    print("  📁 Graph data: D:\\PROJECT\\Machine Learning\\IOT\\graph_data")
    print("  📁 Models: D:\\PROJECT\\Machine Learning\\IOT\\models")
    print("  📁 Results: D:\\PROJECT\\Machine Learning\\IOT\\results")
    print()
    print("Bước tiếp theo:")
    print("  1. Kiểm tra kết quả trong thư mục 'results/'")
    print("  2. Chạy inference: python inference.py")
    print("  3. Để train với full data, chạy: python run_pipeline.py")
    print()
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("❌ LỖI XẢY RA")
    print("=" * 80)
    print(f"Error: {e}")
    print()

    import traceback
    traceback.print_exc()

    print()
    print("Gợi ý khắc phục:")
    print("  1. Kiểm tra pandas đã được cài đặt: pip install pandas")
    print("  2. Kiểm tra PyTorch đã được cài đặt: pip install torch")
    print("  3. Kiểm tra PyTorch Geometric: pip install torch-geometric")
    print("  4. Kiểm tra đường dẫn data directory")
    print()


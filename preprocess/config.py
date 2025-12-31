# Preprocessing Configuration

# Data paths
INPUT_DIR = "dataset-raw"
OUTPUT_DIR = "dataset-processed"

# Files to use
FILES_TO_USE = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
]

# File dùng cho IP_GNN (có cột IP)
IP_GNN_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

# Files to skip (keep for attack stream)
FILES_TO_SKIP = [
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
]

# Processing
CHUNK_SIZE = 100000
MISSING_THRESHOLD = 0.5
CLIP_QUANTILES = [0.01, 0.99]

DROP_COLS = ["Flow ID", "Src Port", "Dst Port", "Timestamp"]

IP_COLS = ["Src IP", "Dst IP"]

# Sampling & Split
SAMPLE_SIZE = 2_000_000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
TARGET_ATTACK_RATIO = 0.30
SEED = 42

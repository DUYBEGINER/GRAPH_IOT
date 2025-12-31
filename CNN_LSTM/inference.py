"""
Inference script for testing trained models locally
Uses the reserved test file: Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
"""
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "CICIDS2018-CSV")
TEST_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

COLS_TO_DROP = [
    'Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port',
    'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'CWE Flag Count'
]

def load_test_data(model_dir, max_samples=100000):
    """Load and preprocess test data"""
    print("Loading test data...")

    # Load scaler and feature names
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)

    # Load test file
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df = pd.read_csv(test_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # Clean
    if 'Label' in df.columns:
        df = df[df['Label'] != 'Label'].copy()

    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    for col in df.columns:
        if col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Create labels
    df['binary_label'] = (df['Label'] != 'Benign').astype(int)

    # Sample if too large
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"  Sampled to {len(df):,} rows")

    # Extract features
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        print(f"  Warning: {len(missing_cols)} missing features")
        for c in missing_cols:
            df[c] = 0

    X = df[feature_names].values
    y = df['binary_label'].values

    # Normalize
    X = scaler.transform(X)

    print(f"  Benign: {(y==0).sum():,}, Attack: {(y==1).sum():,}")
    return X, y, feature_names

def evaluate_cnn(model_dir, X, y):
    """Evaluate CNN model"""
    print("\nEvaluating CNN...")

    model_path = os.path.join(model_dir, 'cnn_model.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'best_model.keras')

    model = tf.keras.models.load_model(model_path)

    # Reshape for CNN
    X_cnn = X.reshape(-1, X.shape[1], 1)

    # Predict
    start_time = time.time()
    y_pred_prob = model.predict(X_cnn, verbose=0)
    latency = (time.time() - start_time) * 1000 / len(X)

    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    return compute_metrics(y, y_pred, y_pred_prob.flatten(), latency, "CNN")

def evaluate_lstm(model_dir, X, y, seq_length=10):
    """Evaluate LSTM model"""
    print("\nEvaluating LSTM...")

    model_path = os.path.join(model_dir, 'lstm_model.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'best_model.keras')

    model = tf.keras.models.load_model(model_path)

    # Create sequences
    n_samples = len(X) - seq_length + 1
    X_seq = np.zeros((n_samples, seq_length, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        X_seq[i] = X[i:i+seq_length]
        y_seq[i] = y[i+seq_length-1]

    # Predict
    start_time = time.time()
    y_pred_prob = model.predict(X_seq, verbose=0)
    latency = (time.time() - start_time) * 1000 / len(X_seq)

    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    return compute_metrics(y_seq, y_pred, y_pred_prob.flatten(), latency, "LSTM")

def compute_metrics(y_true, y_pred, y_pred_prob, latency, model_name):
    """Compute and display metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    print(f"  {model_name} Results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    AUC:       {auc:.4f}")
    print(f"    Latency:   {latency:.4f} ms/sample")
    print(f"    Confusion Matrix:")
    print(f"      TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"      FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc': auc,
        'latency_ms': latency,
        'confusion_matrix': cm.tolist()
    }

def compare_models(cnn_results, lstm_results):
    """Display comparison table"""
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<15} {'CNN':>12} {'LSTM':>12} {'Better':>10}")
    print("-" * 50)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    for metric in metrics:
        cnn_val = cnn_results[metric]
        lstm_val = lstm_results[metric]
        better = "CNN" if cnn_val > lstm_val else "LSTM" if lstm_val > cnn_val else "Tie"
        print(f"{metric.upper():<15} {cnn_val:>12.4f} {lstm_val:>12.4f} {better:>10}")

    # Latency (lower is better)
    cnn_lat = cnn_results['latency_ms']
    lstm_lat = lstm_results['latency_ms']
    better = "CNN" if cnn_lat < lstm_lat else "LSTM"
    print(f"{'LATENCY (ms)':<15} {cnn_lat:>12.4f} {lstm_lat:>12.4f} {better:>10}")
    print("=" * 50)

def main():
    print("=" * 50)
    print("MODEL INFERENCE - LOCAL TEST")
    print("=" * 50)

    # Look for model directories
    cnn_dir = os.path.join(BASE_DIR, 'cnn_output')
    lstm_dir = os.path.join(BASE_DIR, 'lstm_output')
    processed_dir = os.path.join(BASE_DIR, 'processed_data')

    # Check if models exist
    cnn_exists = os.path.exists(cnn_dir) and (
        os.path.exists(os.path.join(cnn_dir, 'cnn_model.keras')) or
        os.path.exists(os.path.join(cnn_dir, 'best_model.keras'))
    )
    lstm_exists = os.path.exists(lstm_dir) and (
        os.path.exists(os.path.join(lstm_dir, 'lstm_model.keras')) or
        os.path.exists(os.path.join(lstm_dir, 'best_model.keras'))
    )

    if not cnn_exists and not lstm_exists:
        print("No trained models found!")
        print(f"Expected CNN model in: {cnn_dir}")
        print(f"Expected LSTM model in: {lstm_dir}")
        return

    # Load test data (use processed_data directory for scaler)
    if not os.path.exists(processed_dir):
        print(f"Processed data not found: {processed_dir}")
        print("Please copy processed_data folder from Kaggle output")
        return

    X, y, feature_names = load_test_data(processed_dir)

    results = []

    if cnn_exists:
        cnn_results = evaluate_cnn(cnn_dir, X, y)
        results.append(cnn_results)

    if lstm_exists:
        lstm_results = evaluate_lstm(lstm_dir, X, y)
        results.append(lstm_results)

    if len(results) == 2:
        compare_models(results[0], results[1])

    # Save results
    results_path = os.path.join(BASE_DIR, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()


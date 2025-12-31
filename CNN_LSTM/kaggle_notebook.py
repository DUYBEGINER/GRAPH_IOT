"""
Complete Kaggle Notebook: CNN + LSTM Training Pipeline
Dataset: CICIDS2018 Binary Classification
Run all cells in order on Kaggle with GPU enabled
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import json
import time
import shutil
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
                                      BatchNormalization, Input, LSTM)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = "/kaggle/input/cicids2018"
OUTPUT_DIR = "/kaggle/working"
EXCLUDED_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

COLS_TO_DROP = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port',
                'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd URG Flags', 'CWE Flag Count']

TOTAL_SAMPLES = 1000000
BENIGN_RATIO = 0.7
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT = 0.3
PATIENCE = 10
RANDOM_STATE = 42
LSTM_SEQ_LENGTH = 10
LSTM_UNITS = 128

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU: {len(gpus)} device(s)")
else:
    print("CPU mode")

try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except:
    pass

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess():
    print("=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)

    # Load data
    print("Loading data...")
    csv_files = sorted(Path(DATA_DIR).glob("*_TrafficForML_CICFlowMeter.csv"))
    csv_files = [f for f in csv_files if f.name != EXCLUDED_FILE]

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False, encoding='utf-8')
        except:
            df = pd.read_csv(f, low_memory=False, encoding='latin-1')
        if 'Label' in df.columns:
            df = df[df['Label'] != 'Label'].copy()
        print(f"  {f.name}: {len(df):,}")
        dfs.append(df)
        gc.collect()

    data = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    print(f"Total: {len(data):,}")

    # Clean
    print("Cleaning...")
    cols_drop = [c for c in COLS_TO_DROP if c in data.columns]
    data = data.drop(columns=cols_drop)

    for col in data.columns:
        if col != 'Label':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    data = data.drop_duplicates()
    gc.collect()

    # Balance
    print("Balancing...")
    data['binary_label'] = (data['Label'] != 'Benign').astype(int)

    benign = data[data['binary_label'] == 0]
    attack = data[data['binary_label'] == 1]

    target_attack = min(int(TOTAL_SAMPLES * (1 - BENIGN_RATIO)), len(attack))
    target_benign = min(int(target_attack * BENIGN_RATIO / (1 - BENIGN_RATIO)), len(benign))

    data = pd.concat([
        benign.sample(n=target_benign, random_state=RANDOM_STATE),
        attack.sample(n=target_attack, random_state=RANDOM_STATE)
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    del benign, attack
    gc.collect()
    print(f"Sampled: {len(data):,}")

    # Features
    print("Extracting features...")
    feature_cols = [c for c in data.columns if c not in ['Label', 'binary_label']]
    feature_cols = [c for c in feature_cols if data[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    variances = data[feature_cols].var()
    feature_cols = variances[variances > 0].index.tolist()
    print(f"Features: {len(feature_cols)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(data[feature_cols])
    y = data['binary_label'].values

    del data
    gc.collect()

    # Split
    print("Splitting...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_RATIO/(1-TEST_RATIO), stratify=y_temp, random_state=RANDOM_STATE)

    del X_temp, y_temp, X, y
    gc.collect()

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Save
    out_dir = os.path.join(OUTPUT_DIR, "processed_data")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'X_train.npy'), X_train.astype(np.float32))
    np.save(os.path.join(out_dir, 'X_val.npy'), X_val.astype(np.float32))
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test.astype(np.float32))
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train.astype(np.int32))
    np.save(os.path.join(out_dir, 'y_val.npy'), y_val.astype(np.int32))
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test.astype(np.int32))

    with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_cols, f)

    print("Preprocessing done!")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

# =============================================================================
# CNN
# =============================================================================
def train_cnn(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "=" * 50)
    print("CNN TRAINING")
    print("=" * 50)

    model_dir = os.path.join(OUTPUT_DIR, "cnn_output")
    os.makedirs(model_dir, exist_ok=True)

    # Reshape
    X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
    X_val_cnn = X_val.reshape(-1, X_val.shape[1], 1)
    X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)

    # Build
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(32, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        BatchNormalization(),
        Dropout(DROPOUT),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(DROPOUT),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])

    print(f"Params: {model.count_params():,}")

    # Train
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]

    start = time.time()
    history = model.fit(X_train_cnn, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(X_val_cnn, y_val), callbacks=callbacks, verbose=1)
    train_time = time.time() - start

    # Evaluate
    start = time.time()
    y_pred_prob = model.predict(X_test_cnn, verbose=0)
    latency = (time.time() - start) * 1000 / len(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    results = compute_metrics(y_test, y_pred, y_pred_prob.flatten(), latency, train_time, history)

    # Save
    model.save(os.path.join(model_dir, 'cnn_model.keras'))
    save_results(model_dir, results, history, "CNN")

    del X_train_cnn, X_val_cnn, X_test_cnn
    gc.collect()

    return results

# =============================================================================
# LSTM
# =============================================================================
def create_sequences(X, y, seq_len):
    n = len(X) - seq_len + 1
    X_seq = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.int32)
    for i in range(n):
        X_seq[i] = X[i:i+seq_len]
        y_seq[i] = y[i+seq_len-1]
    return X_seq, y_seq

def train_lstm(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "=" * 50)
    print("LSTM TRAINING")
    print("=" * 50)

    model_dir = os.path.join(OUTPUT_DIR, "lstm_output")
    os.makedirs(model_dir, exist_ok=True)

    # Create sequences
    print("Creating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, LSTM_SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, LSTM_SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, LSTM_SEQ_LENGTH)
    print(f"Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")

    # Build
    model = Sequential([
        Input(shape=(LSTM_SEQ_LENGTH, X_train.shape[1])),
        LSTM(LSTM_UNITS, return_sequences=True),
        BatchNormalization(),
        Dropout(DROPOUT),
        LSTM(LSTM_UNITS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT),
        Dense(64, activation='relu'),
        Dropout(DROPOUT),
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])

    print(f"Params: {model.count_params():,}")

    # tf.data
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq.astype(np.float32)))
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_seq, y_val_seq.astype(np.float32)))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_seq, y_test_seq.astype(np.float32)))
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Train
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
    train_time = time.time() - start

    # Evaluate
    start = time.time()
    y_pred_prob = model.predict(test_ds, verbose=0)
    latency = (time.time() - start) * 1000 / len(X_test_seq)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    results = compute_metrics(y_test_seq, y_pred, y_pred_prob.flatten(), latency, train_time, history)
    results['sequence_length'] = LSTM_SEQ_LENGTH

    # Save
    model.save(os.path.join(model_dir, 'lstm_model.keras'))
    save_results(model_dir, results, history, "LSTM")

    del X_train_seq, X_val_seq, X_test_seq
    gc.collect()

    return results

# =============================================================================
# UTILS
# =============================================================================
def compute_metrics(y_true, y_pred, y_prob, latency, train_time, history):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}, AUC: {auc:.4f}, Latency: {latency:.4f} ms")

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'auc': float(auc),
        'latency_ms': float(latency),
        'training_time_sec': float(train_time),
        'epochs_trained': len(history.history['loss']),
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat()
    }

def save_results(model_dir, results, history, model_name):
    # History
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f)

    # Results
    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history_dict['loss'], label='Train')
        axes[0].plot(history_dict['val_loss'], label='Val')
        axes[0].set_title(f'{model_name} Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history_dict['accuracy'], label='Train')
        axes[1].plot(history_dict['val_accuracy'], label='Val')
        axes[1].set_title(f'{model_name} Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        cm = np.array(results['confusion_matrix'])
        axes[2].imshow(cm, cmap='Blues')
        axes[2].set_xticks([0, 1])
        axes[2].set_yticks([0, 1])
        axes[2].set_xticklabels(['Benign', 'Attack'])
        axes[2].set_yticklabels(['Benign', 'Attack'])
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title(f'{model_name} Confusion Matrix')
        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                           color='white' if cm[i,j] > cm.max()/2 else 'black')

        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_plots.png'), dpi=150)
        plt.close()
    except:
        pass

def compare_and_save(cnn_results, lstm_results):
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<15} {'CNN':>12} {'LSTM':>12} {'Better':>10}")
    print("-" * 50)

    for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        c, l = cnn_results[m], lstm_results[m]
        better = "CNN" if c > l else "LSTM" if l > c else "Tie"
        print(f"{m.upper():<15} {c:>12.4f} {l:>12.4f} {better:>10}")

    c_lat, l_lat = cnn_results['latency_ms'], lstm_results['latency_ms']
    better = "CNN" if c_lat < l_lat else "LSTM"
    print(f"{'LATENCY (ms)':<15} {c_lat:>12.4f} {l_lat:>12.4f} {better:>10}")
    print("=" * 50)

    # Save comparison
    comparison = {'cnn': cnn_results, 'lstm': lstm_results}
    with open(os.path.join(OUTPUT_DIR, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

def create_zip():
    print("\nCreating zip files...")
    shutil.make_archive(os.path.join(OUTPUT_DIR, 'cnn_output'), 'zip',
                       os.path.join(OUTPUT_DIR, 'cnn_output'))
    shutil.make_archive(os.path.join(OUTPUT_DIR, 'lstm_output'), 'zip',
                       os.path.join(OUTPUT_DIR, 'lstm_output'))
    shutil.make_archive(os.path.join(OUTPUT_DIR, 'processed_data'), 'zip',
                       os.path.join(OUTPUT_DIR, 'processed_data'))
    print("Zip files created!")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 50)
    print("CNN + LSTM TRAINING PIPELINE")
    print("CICIDS2018 Binary Classification")
    print("=" * 50)

    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess()

    # Train CNN
    cnn_results = train_cnn(X_train, X_val, X_test, y_train, y_val, y_test)

    # Train LSTM
    lstm_results = train_lstm(X_train, X_val, X_test, y_train, y_val, y_test)

    # Compare
    compare_and_save(cnn_results, lstm_results)

    # Zip for download
    create_zip()

    print("\n" + "=" * 50)
    print("COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()

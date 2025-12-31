"""
LSTM Training for CICIDS2018 Binary Classification
Designed for Kaggle notebook with GPU
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import json
import time
import shutil
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from config import *

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU: {len(gpus)} device(s)")
else:
    print("Running on CPU")

# Enable mixed precision for faster training
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except:
    pass

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
MODEL_DIR = os.path.join(OUTPUT_DIR, "lstm_output")

def create_sequences(X, y, seq_length):
    """Create sequences for LSTM"""
    n_samples = len(X) - seq_length + 1
    n_features = X.shape[1]

    X_seq = np.zeros((n_samples, seq_length, n_features), dtype=np.float32)
    y_seq = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        X_seq[i] = X[i:i+seq_length]
        y_seq[i] = y[i+seq_length-1]

    return X_seq, y_seq

def load_data():
    """Load and prepare data for LSTM"""
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    print("Creating sequences...")
    X_train, y_train = create_sequences(X_train, y_train, LSTM_SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(X_val, y_val, LSTM_SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(X_test, y_test, LSTM_SEQUENCE_LENGTH)

    gc.collect()
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        Input(shape=input_shape),
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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def make_dataset(X, y, batch_size, shuffle=False):
    """Create tf.data.Dataset for efficient GPU training"""
    ds = tf.data.Dataset.from_tensor_slices((X, y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train(model, X_train, y_train, X_val, y_val):
    """Train the model"""
    print("Training...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.keras'),
                       monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time

    print(f"  Training time: {train_time/60:.2f} min")
    return history, train_time

def evaluate(model, X_test, y_test):
    """Evaluate model and compute metrics"""
    print("Evaluating...")

    test_ds = make_dataset(X_test, y_test, BATCH_SIZE)

    # Measure latency
    start_time = time.time()
    y_pred_prob = model.predict(test_ds, verbose=0)
    latency = (time.time() - start_time) * 1000 / len(X_test)

    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = y_test.flatten()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_prob.flatten())
    except:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    results = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'auc': float(auc),
        'latency_ms': float(latency),
        'confusion_matrix': cm.tolist()
    }

    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  AUC: {auc:.4f}, Latency: {latency:.4f} ms/sample")

    return results

def save_results(model, history, results, train_time):
    """Save model and results"""
    print("Saving results...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))

    # Save history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f)

    # Save results
    results['training_time_sec'] = float(train_time)
    results['epochs_trained'] = len(history.history['loss'])
    results['timestamp'] = datetime.now().isoformat()
    results['sequence_length'] = LSTM_SEQUENCE_LENGTH

    with open(os.path.join(MODEL_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(history_dict['loss'], label='Train')
        axes[0].plot(history_dict['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(history_dict['accuracy'], label='Train')
        axes[1].plot(history_dict['val_accuracy'], label='Val')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Confusion Matrix
        cm = np.array(results['confusion_matrix'])
        im = axes[2].imshow(cm, cmap='Blues')
        axes[2].set_xticks([0, 1])
        axes[2].set_yticks([0, 1])
        axes[2].set_xticklabels(['Benign', 'Attack'])
        axes[2].set_yticklabels(['Benign', 'Attack'])
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title('Confusion Matrix')
        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                           color='white' if cm[i,j] > cm.max()/2 else 'black')

        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'training_plots.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  Warning: Could not save plots - {e}")

    print(f"  Saved to {MODEL_DIR}")

def create_zip():
    """Create zip file for download"""
    if IS_KAGGLE:
        zip_path = "/kaggle/working/lstm_output"
        shutil.make_archive(zip_path, 'zip', MODEL_DIR)
        print(f"Created: {zip_path}.zip")

def main():
    print("=" * 50)
    print("LSTM TRAINING")
    print("=" * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = build_model(input_shape=(LSTM_SEQUENCE_LENGTH, X_train.shape[2]))
    print(f"Model params: {model.count_params():,}")

    history, train_time = train(model, X_train, y_train, X_val, y_val)

    results = evaluate(model, X_test, y_test)

    save_results(model, history, results, train_time)

    create_zip()

    print("\n" + "=" * 50)
    print("LSTM TRAINING COMPLETE")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()


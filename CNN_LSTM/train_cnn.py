"""
CNN Training for CICIDS2018 Binary Classification
Designed for Kaggle notebook with GPU
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import shutil
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense,
                                      Dropout, BatchNormalization, Input)
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

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
MODEL_DIR = os.path.join(OUTPUT_DIR, "cnn_output")

def load_data():
    """Load preprocessed data"""
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    # Reshape for CNN: (samples, features, 1)
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_val = X_val.reshape(-1, X_val.shape[1], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], 1)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape):
    """Build CNN model"""
    model = Sequential([
        Input(shape=input_shape),
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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train(model, X_train, y_train, X_val, y_val):
    """Train the model"""
    print("Training...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.keras'),
                       monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time

    print(f"  Training time: {train_time/60:.2f} min")
    return history, train_time

def evaluate(model, X_test, y_test):
    """Evaluate model and compute metrics"""
    print("Evaluating...")

    # Measure latency
    start_time = time.time()
    y_pred_prob = model.predict(X_test, verbose=0)
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
    model.save(os.path.join(MODEL_DIR, 'cnn_model.keras'))

    # Save history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f)

    # Save results
    results['training_time_sec'] = float(train_time)
    results['epochs_trained'] = len(history.history['loss'])
    results['timestamp'] = datetime.now().isoformat()

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
        zip_path = "/kaggle/working/cnn_output"
        shutil.make_archive(zip_path, 'zip', MODEL_DIR)
        print(f"Created: {zip_path}.zip")

def main():
    print("=" * 50)
    print("CNN TRAINING")
    print("=" * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = build_model(input_shape=(X_train.shape[1], 1))
    print(f"Model params: {model.count_params():,}")

    history, train_time = train(model, X_train, y_train, X_val, y_val)

    results = evaluate(model, X_test, y_test)

    save_results(model, history, results, train_time)

    create_zip()

    print("\n" + "=" * 50)
    print("CNN TRAINING COMPLETE")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()


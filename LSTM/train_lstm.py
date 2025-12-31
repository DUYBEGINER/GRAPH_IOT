import os

# ============================================================================
# 1. FIX: TẮT LOG CẢNH BÁO ĐỎ CỦA TENSORFLOW
# ============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    average_precision_score
import json

# ============================================================================
# 2. CẤU HÌNH GPU & MIXED PRECISION
# ============================================================================
try:
    from tensorflow.keras import mixed_precision

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print('✓ Mixed precision policy set to mixed_float16')
except Exception as e:
    print(f'⚠️ Mixed precision not supported: {e}')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'✓ Found {len(gpus)} GPU(s). Memory growth enabled.')
    except RuntimeError as e:
        print(e)

# ============================================================================
# 3. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================================
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    print("🌍 ENVIRONMENT: KAGGLE DETECTED")
    INPUT_DIR = "/kaggle/working/processed_lstm"
    MODEL_DIR = "/kaggle/working/models"
    BATCH_SIZE = 2048
else:
    print("💻 ENVIRONMENT: LOCAL DESKTOP DETECTED")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, 'processed_lstm')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    BATCH_SIZE = 1024

# Hyperparameters
EPOCHS = 30
LEARNING_RATE = 0.001
LSTM_UNITS = 128
DROPOUT_RATE = 0.3


# ============================================================================
# 4. CÁC HÀM HỖ TRỢ
# ============================================================================
def load_data_as_dataset(X_path, y_path, is_training=True):
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {X_path}")

    X = np.load(X_path)
    y = np.load(y_path).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, X.shape


def build_model(input_shape):
    print(f"\nBuilding Optimized LSTM Model with input shape: {input_shape}")
    model = Sequential([
        Input(shape=input_shape),
        LSTM(LSTM_UNITS, return_sequences=True),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model


def plot_training_history(history):
    """Vẽ và HIỂN THỊ biểu đồ huấn luyện"""
    print("\n📊 Plotting training history...")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training vs Validation Accuracy')
    plt.grid(True, alpha=0.3)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training vs Validation Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Lưu trước rồi mới show
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()  # <--- QUAN TRỌNG: Hiển thị lên màn hình
    print("✓ Training history displayed.")


def plot_evaluation_metrics(model, test_ds, y_test_path):
    """Vẽ và HIỂN THỊ Confusion Matrix, ROC, PR Curve"""
    print("\n📊 Generating detailed evaluation metrics...")

    y_true = np.load(y_test_path)
    y_pred_prob = model.predict(test_ds, verbose=1).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    plt.figure(figsize=(18, 5))

    # 1. Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 2. ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    plt.plot(recall, precision, color='green', lw=2, label=f'AP = {avg_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'evaluation_metrics.png'))
    plt.show()  # <--- QUAN TRỌNG: Hiển thị lên màn hình

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Attack'])
    print(report)
    with open(os.path.join(MODEL_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def main():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    # 1. Load Data
    print("⏳ Loading data...")
    train_ds, train_shape = load_data_as_dataset(
        os.path.join(INPUT_DIR, 'X_train.npy'),
        os.path.join(INPUT_DIR, 'y_train.npy'),
        is_training=True
    )
    val_ds, _ = load_data_as_dataset(
        os.path.join(INPUT_DIR, 'X_val.npy'),
        os.path.join(INPUT_DIR, 'y_val.npy'),
        is_training=False
    )
    test_ds, _ = load_data_as_dataset(
        os.path.join(INPUT_DIR, 'X_test.npy'),
        os.path.join(INPUT_DIR, 'y_test.npy'),
        is_training=False
    )

    # 2. Train
    model = build_model((train_shape[1], train_shape[2]))

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(os.path.join(MODEL_DIR, 'best_lstm_model.keras'), save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    print("\n🚀 Starting Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # 3. Visualize Training History
    plot_training_history(history)

    # 4. Evaluate & Visualize Metrics
    print("\n🔍 Final Evaluation...")
    eval_results = model.evaluate(test_ds)
    print(f"Test Accuracy: {eval_results[1]:.4f}")

    plot_evaluation_metrics(model, test_ds, os.path.join(INPUT_DIR, 'y_test.npy'))

    # Save Info
    results = {
        'test_accuracy': float(eval_results[1]),
        'test_precision': float(eval_results[2]),
        'test_recall': float(eval_results[3]),
        'config': {'batch_size': BATCH_SIZE, 'lstm_units': LSTM_UNITS}
    }
    with open(os.path.join(MODEL_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ All Done! Charts are displayed above.")


if __name__ == "__main__":
    main()
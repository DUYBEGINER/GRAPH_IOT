import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json


try:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print('✓ Mixed precision policy set to mixed_float16')
except Exception as e:
    print(f'⚠️ Mixed precision not supported: {e}')

# 2. Cấu hình GPU Memory Growth (không cho TF chiếm toàn bộ VRAM ngay lập tức)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'✓ Found {len(gpus)} GPU(s). Memory growth enabled.')
    except RuntimeError as e:
        print(e)

# ============================================================================ 
# CONFIGURATION (AUTO-DETECT KAGGLE VS LOCAL)
# ============================================================================ 
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    print("🌍 ENVIRONMENT: KAGGLE DETECTED")
    INPUT_DIR = "/kaggle/working/processed_lstm"
    MODEL_DIR = "/kaggle/working/models"
    # Kaggle T4 x2 GPU rất mạnh, tăng batch size để train nhanh hơn
    BATCH_SIZE = 2048 
    print(f"   - Input: {INPUT_DIR}")
    print(f"   - Model: {MODEL_DIR}")
    print(f"   - Batch: {BATCH_SIZE}")
else:
    print("💻 ENVIRONMENT: LOCAL DESKTOP DETECTED")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, 'processed_lstm')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    BATCH_SIZE = 1024 
    print(f"   - Input: {INPUT_DIR}")
    print(f"   - Model: {MODEL_DIR}")
    print(f"   - Batch: {BATCH_SIZE}")

EPOCHS = 30
LEARNING_RATE = 0.001
LSTM_UNITS = 128
DROPOUT_RATE = 0.3

def load_data_as_dataset(X_path, y_path, is_training=True):
    """
    Sử dụng tf.data.Dataset để tối ưu hóa việc nạp dữ liệu vào GPU.
    Đây là cách tốt nhất để đảm bảo GPU không bị 'đói' dữ liệu.
    """
    X = np.load(X_path)
    y = np.load(y_path).astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
        
    dataset = dataset.batch(BATCH_SIZE)
    # Prefetch: CPU chuẩn bị batch tiếp theo trong khi GPU đang xử lý batch hiện tại
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, X.shape

def build_model(input_shape):
    """Xây dựng kiến trúc LSTM Model tối ưu cho GPU"""
    print(f"\nBuilding Optimized LSTM Model with input shape: {input_shape}")
    
    model = Sequential([
        Input(shape=input_shape),
        
        # LSTM Layer 1
        LSTM(LSTM_UNITS, return_sequences=True),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        # LSTM Layer 2
        LSTM(LSTM_UNITS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        # Dense Layers
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        
        # Output Layer (Dùng float32 để đảm bảo độ chính xác ở lớp cuối cùng)
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    # Optimizer tối ưu cho Deep Learning
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    model.summary()
    return model

def plot_evaluation_metrics(model, test_ds, y_test_path):
    """Vẽ Confusion Matrix và ROC Curve"""
    print("\nGenerating evaluation plots...")
    
    # Load y_test true values (vì test_ds đã bị batch, khó lấy trực tiếp)
    y_true = np.load(y_test_path)
    
    # Predict (Lấy xác suất)
    y_pred_prob = model.predict(test_ds, verbose=1)
    # Convert to binary
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # 2. Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Attack'])
    print(report)
    with open(os.path.join(MODEL_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MODEL_DIR, 'roc_curve.png'))
    plt.close()
    
    print(f"✓ Plots saved to {MODEL_DIR}")

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # 1. Load Data bằng tf.data pipeline
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
    
    # 2. Build Model
    # input_shape = (TimeSteps, Features)
    input_dim = (train_shape[1], train_shape[2])
    model = build_model(input_dim)
    
    # 3. Callbacks thông minh
    checkpoint_path = os.path.join(MODEL_DIR, 'best_lstm_model.keras')
    callbacks = [
        # Dừng sớm nếu không còn cải thiện (tiết kiệm thời gian)
        EarlyStopping(patience=7, restore_best_weights=True, monitor='val_loss'),
        # Lưu lại model tốt nhất
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
        # Tự động giảm Learning Rate nếu model bị chững lại (giúp hội tụ sâu hơn)
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    ]
    
    # 4. Train
    print("\n🚀 Starting GPU-Accelerated Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluate
    print("\n🔍 Final Evaluation on Test set...")
    eval_results = model.evaluate(test_ds)
    print(f"Test Accuracy: {eval_results[1]:.4f}")
    
    # 6. Advanced Evaluation Plots
    plot_evaluation_metrics(model, test_ds, os.path.join(INPUT_DIR, 'y_test.npy'))
    
    # 7. Save Results
    results = {
        'test_accuracy': float(eval_results[1]),
        'test_precision': float(eval_results[2]),
        'test_recall': float(eval_results[3]),
        'config': {
            'batch_size': BATCH_SIZE,
            'lstm_units': LSTM_UNITS,
            'mixed_precision': True
        }
    }
    
    with open(os.path.join(MODEL_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Training completed! Model saved in: {MODEL_DIR}")

if __name__ == "__main__":
    main()
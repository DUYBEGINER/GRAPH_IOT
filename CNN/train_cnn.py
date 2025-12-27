"""
======================================================================================
HUẤN LUYỆN MÔ HÌNH CNN - PHÁT HIỆN LƯU LƯỢNG MẠNG IOT BẤT THƯỜNG
======================================================================================

Script này huấn luyện mô hình CNN 1D cho bài toán phân loại binary:
- Benign (0): Lưu lượng mạng bình thường
- Attack (1): Lưu lượng mạng bất thường/tấn công

Mô hình CNN 1D được thiết kế để học các patterns từ network flow features.

Có thể chạy trên cả Kaggle và Local.
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KIỂM TRA MÔI TRƯỜNG VÀ IMPORT THƯ VIỆN
# ============================================================================

# Kiểm tra môi trường chạy (Kaggle hoặc Local)
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    print("🌐 Đang chạy trên KAGGLE")
else:
    print("💻 Đang chạy trên LOCAL")

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ Lỗi: TensorFlow chưa được cài đặt!")
    print("   Cài đặt bằng: pip install tensorflow")
    sys.exit(1)

# Import sklearn cho metrics
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

# ============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================================

if IS_KAGGLE:
    # Đường dẫn trên Kaggle
    PROCESSED_DATA_DIR = "/kaggle/working/processed_data_cnn"
    OUTPUT_DIR = "/kaggle/working/cnn_results"
else:
    # Đường dẫn Local
    PROCESSED_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\processed_data_cnn"
    OUTPUT_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\results"

# ============================================================================
# CẤU HÌNH MÔ HÌNH VÀ HUẤN LUYỆN
# ============================================================================

# Hyperparameters cho CNN
CNN_CONFIG = {
    # Kiến trúc mạng
    'conv_filters': [64, 128, 256],      # Số filters cho mỗi Conv layer
    'kernel_size': 3,                     # Kích thước kernel
    'pool_size': 2,                       # Kích thước pooling
    'dense_units': [128, 64],             # Số units cho Dense layers
    'dropout_rate': 0.3,                  # Tỷ lệ dropout

    # Huấn luyện
    'batch_size': 256,                    # Batch size
    'epochs': 30,                         # Số epochs tối đa
    'learning_rate': 0.001,               # Learning rate
    'early_stopping_patience': 10,        # Patience cho early stopping
    'reduce_lr_patience': 5,              # Patience cho reduce LR
    'reduce_lr_factor': 0.5,              # Factor giảm LR
    'min_lr': 1e-7,                       # LR tối thiểu

    # Class weights để xử lý imbalanced data
    'use_class_weight': True,             # Sử dụng class weight
}

# ============================================================================
# LOAD DỮ LIỆU ĐÃ XỬ LÝ
# ============================================================================

def load_processed_data(data_dir):
    """
    Load dữ liệu đã được tiền xử lý

    Args:
        data_dir: Đường dẫn thư mục chứa dữ liệu

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, metadata
    """
    data_dir = Path(data_dir)

    print("\n" + "="*80)
    print("📂 ĐANG LOAD DỮ LIỆU ĐÃ XỬ LÝ")
    print("="*80)

    # Load numpy arrays
    X_train = np.load(data_dir / 'X_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')

    print(f"   X_train: {X_train.shape}")
    print(f"   X_val:   {X_val.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_val:   {y_val.shape}")
    print(f"   y_test:  {y_test.shape}")

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"\n   Số features: {metadata['n_features']}")
    print(f"   Train samples: {metadata['train_samples']:,}")
    print(f"   Val samples: {metadata['val_samples']:,}")
    print(f"   Test samples: {metadata['test_samples']:,}")

    return X_train, X_val, X_test, y_train, y_val, y_test, metadata


# ============================================================================
# XÂY DỰNG MÔ HÌNH CNN
# ============================================================================

def build_cnn_model(input_shape, config=CNN_CONFIG):
    """
    Xây dựng mô hình CNN 1D cho phân loại binary

    Kiến trúc:
    - Nhiều block Conv1D + BatchNorm + ReLU + MaxPooling + Dropout
    - Flatten
    - Dense layers với Dropout
    - Output layer với Sigmoid

    Args:
        input_shape: Shape của input (n_features, 1)
        config: Dictionary chứa các hyperparameters

    Returns:
        model: Mô hình Keras đã compile
    """
    print("\n" + "="*80)
    print("🏗️ ĐANG XÂY DỰNG MÔ HÌNH CNN")
    print("="*80)

    model = models.Sequential(name="CNN_IDS")

    # Input layer
    model.add(layers.InputLayer(input_shape=input_shape))

    # Convolutional blocks
    for i, filters in enumerate(config['conv_filters']):
        # Conv1D layer
        model.add(layers.Conv1D(
            filters=filters,
            kernel_size=config['kernel_size'],
            padding='same',
            name=f'conv1d_{i+1}'
        ))

        # Batch Normalization
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))

        # Activation
        model.add(layers.Activation('relu', name=f'relu_{i+1}'))

        # MaxPooling (chỉ áp dụng nếu kích thước đủ lớn)
        model.add(layers.MaxPooling1D(
            pool_size=config['pool_size'],
            padding='same',
            name=f'maxpool_{i+1}'
        ))

        # Dropout
        model.add(layers.Dropout(
            config['dropout_rate'],
            name=f'dropout_conv_{i+1}'
        ))

    # Flatten
    model.add(layers.Flatten(name='flatten'))

    # Dense layers
    for i, units in enumerate(config['dense_units']):
        model.add(layers.Dense(units, name=f'dense_{i+1}'))
        model.add(layers.BatchNormalization(name=f'bn_dense_{i+1}'))
        model.add(layers.Activation('relu', name=f'relu_dense_{i+1}'))
        model.add(layers.Dropout(config['dropout_rate'], name=f'dropout_dense_{i+1}'))

    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid', name='output'))

    # Compile model
    optimizer = Adam(learning_rate=config['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    # In summary
    print(f"\n   Input shape: {input_shape}")
    print(f"   Conv filters: {config['conv_filters']}")
    print(f"   Dense units: {config['dense_units']}")
    print(f"   Dropout rate: {config['dropout_rate']}")
    print(f"   Learning rate: {config['learning_rate']}")

    model.summary()

    return model


# ============================================================================
# TÍNH CLASS WEIGHTS
# ============================================================================

def compute_class_weights(y_train):
    """
    Tính class weights để xử lý imbalanced data

    Args:
        y_train: Labels của training set

    Returns:
        class_weight: Dictionary {0: weight_0, 1: weight_1}
    """
    # Đếm số lượng mỗi class
    n_benign = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    total = len(y_train)

    # Tính weights
    weight_benign = total / (2 * n_benign)
    weight_attack = total / (2 * n_attack)

    class_weight = {
        0: weight_benign,
        1: weight_attack
    }

    print(f"\n📊 Class weights:")
    print(f"   Benign (0): {n_benign:,} mẫu, weight = {weight_benign:.4f}")
    print(f"   Attack (1): {n_attack:,} mẫu, weight = {weight_attack:.4f}")

    return class_weight


# ============================================================================
# CALLBACKS
# ============================================================================

def get_callbacks(output_dir, config=CNN_CONFIG):
    """
    Tạo các callbacks cho quá trình huấn luyện

    Args:
        output_dir: Đường dẫn lưu model
        config: Dictionary chứa các hyperparameters

    Returns:
        List các callbacks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callback_list = [
        # Early stopping - dừng nếu val_loss không cải thiện
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate khi val_loss không cải thiện
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['reduce_lr_factor'],
            patience=config['reduce_lr_patience'],
            min_lr=config['min_lr'],
            verbose=1
        ),

        # Model checkpoint - lưu model tốt nhất
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.keras'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),

        # TensorBoard logs (optional)
        callbacks.TensorBoard(
            log_dir=str(output_dir / 'logs'),
            histogram_freq=1
        ),

        # CSV logger - lưu history ra file
        callbacks.CSVLogger(
            filename=str(output_dir / 'training_history.csv'),
            separator=',',
            append=False
        )
    ]

    return callback_list


# ============================================================================
# HUẤN LUYỆN MÔ HÌNH
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val,
                output_dir, config=None):
    """
    Huấn luyện mô hình CNN

    Args:
        model: Mô hình Keras
        X_train, y_train: Dữ liệu training
        X_val, y_val: Dữ liệu validation
        output_dir: Đường dẫn lưu kết quả
        config: Dictionary chứa các hyperparameters

    Returns:
        history: History object của quá trình training
    """
    if config is None:
        config = CNN_CONFIG
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")
    print("="*80)

    # Tính class weights nếu cần
    class_weight = None
    if config['use_class_weight']:
        class_weight = compute_class_weights(y_train)

    # Lấy callbacks
    callback_list = get_callbacks(output_dir, config)

    print(f"\n   Batch size: {config['batch_size']}")
    print(f"   Max epochs: {config['epochs']}")
    print(f"   Early stopping patience: {config['early_stopping_patience']}")

    # Huấn luyện
    start_time = datetime.now()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        class_weight=class_weight,
        callbacks=callback_list,
        verbose=1
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print(f"\n⏱️ Thời gian huấn luyện: {training_time/60:.2f} phút")

    return history, training_time


# ============================================================================
# ĐÁNH GIÁ MÔ HÌNH
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình trên test set

    Args:
        model: Mô hình đã train
        X_test, y_test: Dữ liệu test

    Returns:
        results: Dictionary chứa các metrics
    """
    print("\n" + "="*80)
    print("📊 ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SET")
    print("="*80)

    # Dự đoán
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Tính các metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    # In kết quả
    print(f"\n   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")

    # Classification report
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT:")
    print("-"*60)
    target_names = ['Benign', 'Attack']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix
    print("-"*60)
    print("CONFUSION MATRIX:")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"              Benign  Attack")
    print(f"Actual Benign   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Attack   {cm[1,0]:6d}  {cm[1,1]:6d}")

    # Tính detection rate và false alarm rate
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n   Detection Rate (True Positive Rate): {detection_rate:.4f}")
    print(f"   False Alarm Rate (False Positive Rate): {false_alarm_rate:.4f}")

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'confusion_matrix': cm.tolist(),
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred
    }

    return results


# ============================================================================
# VẼ BIỂU ĐỒ
# ============================================================================

def plot_training_history(history, output_dir):
    """
    Vẽ biểu đồ training history

    Args:
        history: History object từ model.fit()
        output_dir: Đường dẫn lưu biểu đồ
    """
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot Precision & Recall
    axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 0].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot AUC
    axes[1, 1].plot(history.history['auc'], label='Train AUC', linewidth=2)
    axes[1, 1].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
    axes[1, 1].set_title('AUC-ROC', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training History - CNN IoT Anomaly Detection',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Đã lưu biểu đồ training history: {output_dir / 'training_history.png'}")


def plot_confusion_matrix(cm, output_dir):
    """
    Vẽ confusion matrix

    Args:
        cm: Confusion matrix
        output_dir: Đường dẫn lưu biểu đồ
    """
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Vẽ heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ['Benign', 'Attack']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='Actual',
           xlabel='Predicted')

    # Thêm text vào các ô
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Đã lưu confusion matrix: {output_dir / 'confusion_matrix.png'}")


def plot_roc_curve(y_test, y_pred_prob, output_dir):
    """
    Vẽ ROC curve

    Args:
        y_test: Labels thực tế
        y_pred_prob: Xác suất dự đoán
        output_dir: Đường dẫn lưu biểu đồ
    """
    output_dir = Path(output_dir)

    # Tính ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='blue', linewidth=2,
            label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Đã lưu ROC curve: {output_dir / 'roc_curve.png'}")


# ============================================================================
# LƯU KẾT QUẢ
# ============================================================================

def save_results(results, history, training_time, output_dir, config):
    """
    Lưu kết quả và thông tin mô hình

    Args:
        results: Dictionary chứa metrics
        history: Training history
        training_time: Thời gian training (giây)
        output_dir: Đường dẫn lưu
        config: Config của mô hình
    """
    output_dir = Path(output_dir)

    # Tạo summary
    summary = {
        'model_name': 'CNN_1D_IDS',
        'task': 'Binary Classification - IoT Anomaly Detection',
        'dataset': 'CICIDS2018',
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'config': config,
        'results': {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'auc_roc': float(results['auc_roc']),
            'detection_rate': float(results['detection_rate']),
            'false_alarm_rate': float(results['false_alarm_rate']),
            'confusion_matrix': results['confusion_matrix']
        },
        'final_epoch': len(history.history['loss']),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_auc': float(max(history.history['val_auc'])),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Lưu summary
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n💾 Đã lưu kết quả: {output_dir / 'results_summary.json'}")


# ============================================================================
# HÀM CHÍNH
# ============================================================================

def main():
    """Hàm chính để chạy toàn bộ pipeline"""

    print("\n" + "="*80)
    print("🔧 HUẤN LUYỆN MÔ HÌNH CNN - PHÁT HIỆN LƯU LƯỢNG MẠNG IOT BẤT THƯỜNG")
    print("="*80)

    # Tạo thư mục output
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bước 1: Load dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_processed_data(PROCESSED_DATA_DIR)

    # Bước 2: Xây dựng mô hình
    input_shape = (X_train.shape[1], X_train.shape[2])  # (n_features, 1)
    model = build_cnn_model(input_shape)

    # Bước 3: Huấn luyện mô hình
    history, training_time = train_model(
        model, X_train, y_train, X_val, y_val,
        output_dir, CNN_CONFIG
    )

    # Bước 4: Đánh giá mô hình
    results = evaluate_model(model, X_test, y_test)

    # Bước 5: Vẽ biểu đồ
    plot_training_history(history, output_dir)
    plot_confusion_matrix(np.array(results['confusion_matrix']), output_dir)
    plot_roc_curve(y_test, results['y_pred_prob'], output_dir)

    # Bước 6: Lưu kết quả
    save_results(results, history, training_time, output_dir, CNN_CONFIG)

    # Lưu model cuối cùng
    model.save(output_dir / 'final_model.keras')
    print(f"💾 Đã lưu model cuối cùng: {output_dir / 'final_model.keras'}")

    print("\n" + "="*80)
    print("✅ HOÀN THÀNH HUẤN LUYỆN MÔ HÌNH CNN!")
    print("="*80)
    print(f"\n📁 Tất cả kết quả được lưu tại: {output_dir}")
    print(f"   - best_model.keras: Model tốt nhất (theo val_auc)")
    print(f"   - final_model.keras: Model cuối cùng")
    print(f"   - training_history.png: Biểu đồ training")
    print(f"   - confusion_matrix.png: Confusion matrix")
    print(f"   - roc_curve.png: ROC curve")
    print(f"   - results_summary.json: Tóm tắt kết quả")

    return model, history, results


# ============================================================================
# CHẠY SCRIPT
# ============================================================================

if __name__ == "__main__":
    model, history, results = main()


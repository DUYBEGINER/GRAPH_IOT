"""
======================================================================================
BƯỚC 3: TRAIN MÔ HÌNH CNN CHO PHÁT HIỆN LƯU LƯỢNG MẠNG IOT BẤT THƯỜNG
======================================================================================

Kiến trúc CNN theo yêu cầu:
- Input Layer: Shape (num_features, 1)
- Conv1D (32 filters, kernel 2) -> MaxPooling1D (2)
- Conv1D (32 filters, kernel 2) -> MaxPooling1D (2)
- Conv1D (64 filters, kernel 2) -> MaxPooling1D (2)
- Conv1D (64 filters, kernel 2) -> MaxPooling1D (2)
- Conv1D (64 filters, kernel 2) -> MaxPooling1D (2)
- BatchNormalization + Dropout (0.5)
- Flatten
- Dense(1, activation='sigmoid')

Loss: binary_crossentropy
Optimizer: Adam
Metrics: Accuracy, Precision, Recall

Có thể chạy trên cả Kaggle và Local
"""

import os
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TENSORFLOW/KERAS
# ============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense,
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.metrics import Precision, Recall

# Kiểm tra GPU
print("="*80)
print("🖥️ THÔNG TIN HỆ THỐNG")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"   - {gpu}")
    # Cấu hình GPU memory growth để tránh chiếm hết bộ nhớ
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ Không có GPU, sẽ sử dụng CPU")

# Kiểm tra môi trường chạy
IS_KAGGLE = os.path.exists('/kaggle/input')

# ============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================================
if IS_KAGGLE:
    TRAINING_DATA_DIR = "/kaggle/working/training_data"
    MODEL_DIR = "/kaggle/working/models"
    LOG_DIR = "/kaggle/working/logs"
    print("🌐 Đang chạy trên KAGGLE")
else:
    TRAINING_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\training_data"
    MODEL_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\models"
    LOG_DIR = r"D:\PROJECT\Machine Learning\IOT\CNN\logs"
    print("💻 Đang chạy trên LOCAL")

# ============================================================================
# CẤU HÌNH HUẤN LUYỆN
# ============================================================================

# Hyperparameters
BATCH_SIZE = 256        # Batch size cho training
EPOCHS = 50             # Số epochs tối đa
LEARNING_RATE = 0.001   # Learning rate ban đầu

# Regularization
DROPOUT_RATE = 0.5      # Dropout rate trước Flatten

# Early stopping
PATIENCE = 10           # Số epochs chờ trước khi dừng

# Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================================
# HÀM XÂY DỰNG MÔ HÌNH CNN
# ============================================================================

def build_cnn_model(input_shape):
    """
    Xây dựng mô hình CNN cho phân loại binary

    Kiến trúc theo yêu cầu:
    - 5 lớp Conv1D với MaxPooling
    - BatchNormalization và Dropout trước Flatten
    - Output layer với sigmoid activation

    Args:
        input_shape: Shape của input (n_features, 1)

    Returns:
        model: Keras Sequential model
    """
    print("\n" + "="*80)
    print("🏗️ ĐANG XÂY DỰNG MÔ HÌNH CNN")
    print("="*80)
    print(f"   Input shape: {input_shape}")

    model = Sequential(name='CNN_Binary_Classification')

    # Input layer
    model.add(Input(shape=input_shape))

    # ========== KHỐI CONV 1 ==========
    # Conv1D (32 filters, kernel 2x1) -> MaxPooling1D (2)
    model.add(Conv1D(
        filters=32,
        kernel_size=2,
        activation='relu',
        padding='same',  # Giữ nguyên kích thước
        name='conv1d_1'
    ))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_1'))

    # ========== KHỐI CONV 2 ==========
    # Conv1D (32 filters, kernel 2x1) -> MaxPooling1D (2)
    model.add(Conv1D(
        filters=32,
        kernel_size=2,
        activation='relu',
        padding='same',
        name='conv1d_2'
    ))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_2'))

    # ========== KHỐI CONV 3 ==========
    # Conv1D (64 filters, kernel 2x1) -> MaxPooling1D (2)
    model.add(Conv1D(
        filters=64,
        kernel_size=2,
        activation='relu',
        padding='same',
        name='conv1d_3'
    ))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_3'))

    # ========== KHỐI CONV 4 ==========
    # Conv1D (64 filters, kernel 2x1) -> MaxPooling1D (2)
    model.add(Conv1D(
        filters=64,
        kernel_size=2,
        activation='relu',
        padding='same',
        name='conv1d_4'
    ))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_4'))

    # ========== KHỐI CONV 5 ==========
    # Conv1D (64 filters, kernel 2x1) -> MaxPooling1D (2)
    model.add(Conv1D(
        filters=64,
        kernel_size=2,
        activation='relu',
        padding='same',
        name='conv1d_5'
    ))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_5'))

    # ========== REGULARIZATION ==========
    # BatchNormalization và Dropout trước Flatten
    model.add(BatchNormalization(name='batch_norm'))
    model.add(Dropout(DROPOUT_RATE, name='dropout'))

    # ========== FLATTEN ==========
    model.add(Flatten(name='flatten'))

    # ========== OUTPUT LAYER ==========
    # Dense(1, activation='sigmoid') cho binary classification
    model.add(Dense(1, activation='sigmoid', name='output'))

    # ========== COMPILE ==========
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )

    # In tóm tắt mô hình
    print("\n   📋 KIẾN TRÚC MÔ HÌNH:")
    model.summary()

    return model


def load_training_data(data_dir):
    """
    Load dữ liệu training đã được chuẩn bị từ step 2

    Args:
        data_dir: Đường dẫn thư mục chứa dữ liệu

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, class_weights
    """
    print("\n" + "="*80)
    print("📂 ĐANG LOAD DỮ LIỆU TRAINING...")
    print("="*80)

    data_dir = Path(data_dir)

    # Load numpy arrays
    X_train = np.load(data_dir / 'X_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    y_test = np.load(data_dir / 'y_test.npy')

    print(f"   ✅ X_train: {X_train.shape}")
    print(f"   ✅ X_val:   {X_val.shape}")
    print(f"   ✅ X_test:  {X_test.shape}")
    print(f"   ✅ y_train: {y_train.shape}")
    print(f"   ✅ y_val:   {y_val.shape}")
    print(f"   ✅ y_test:  {y_test.shape}")

    # Load class weights nếu có
    class_weights = None
    class_weights_path = data_dir / 'class_weights.pkl'
    if class_weights_path.exists():
        with open(class_weights_path, 'rb') as f:
            class_weights = pickle.load(f)
        print(f"\n   ⚖️ Class weights loaded:")
        print(f"      Class 0 (Benign): {class_weights[0]:.4f}")
        print(f"      Class 1 (Attack): {class_weights[1]:.4f}")

    # Thống kê phân bố
    print(f"\n   📊 PHÂN BỐ DỮ LIỆU:")
    for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        benign = (y == 0).sum()
        attack = (y == 1).sum()
        total = len(y)
        print(f"      {name}: Benign={benign:,} ({benign/total*100:.1f}%), Attack={attack:,} ({attack/total*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


def create_callbacks(model_dir, log_dir):
    """
    Tạo các callbacks cho training

    Callbacks:
    - EarlyStopping: Dừng sớm khi val_loss không giảm
    - ModelCheckpoint: Lưu model tốt nhất
    - ReduceLROnPlateau: Giảm learning rate khi plateau
    - TensorBoard: Logging cho visualization
    """
    print("\n📌 ĐANG CẤU HÌNH CALLBACKS...")

    model_dir = Path(model_dir)
    log_dir = Path(log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []

    # 1. Early Stopping
    # Dừng training khi val_loss không cải thiện sau PATIENCE epochs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        verbose=1,
        mode='min',
        restore_best_weights=True  # Khôi phục weights tốt nhất
    )
    callbacks.append(early_stopping)
    print(f"   ✅ EarlyStopping: patience={PATIENCE}")

    # 2. Model Checkpoint
    # Lưu model có val_loss thấp nhất
    checkpoint_path = model_dir / 'best_model.keras'
    model_checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks.append(model_checkpoint)
    print(f"   ✅ ModelCheckpoint: {checkpoint_path}")

    # 3. Reduce Learning Rate on Plateau
    # Giảm LR khi val_loss không giảm
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # Giảm LR còn 1/2
        patience=5,        # Chờ 5 epochs
        min_lr=1e-7,       # LR tối thiểu
        verbose=1
    )
    callbacks.append(reduce_lr)
    print(f"   ✅ ReduceLROnPlateau: factor=0.5, patience=5")

    # 4. TensorBoard (optional)
    tensorboard_log = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(
        log_dir=str(tensorboard_log),
        histogram_freq=1
    )
    callbacks.append(tensorboard)
    print(f"   ✅ TensorBoard: {tensorboard_log}")

    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, class_weights, callbacks):
    """
    Huấn luyện mô hình

    Args:
        model: Keras model
        X_train, y_train: Dữ liệu training
        X_val, y_val: Dữ liệu validation
        class_weights: Dictionary class weights
        callbacks: List các callbacks

    Returns:
        history: Training history
    """
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")
    print("="*80)
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Class weights: {'Có' if class_weights else 'Không'}")

    start_time = datetime.now()

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights,  # Sử dụng class weights để xử lý imbalance
        callbacks=callbacks,
        verbose=1
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print(f"\n   ⏱️ Thời gian training: {training_time/60:.2f} phút")
    print(f"   📈 Best val_loss: {min(history.history['val_loss']):.4f}")
    print(f"   📈 Best val_accuracy: {max(history.history['val_accuracy']):.4f}")

    return history, training_time


def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình trên test set

    Args:
        model: Trained model
        X_test, y_test: Dữ liệu test

    Returns:
        results: Dictionary kết quả đánh giá
    """
    print("\n" + "="*80)
    print("📊 ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SET")
    print("="*80)

    # Evaluate
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=1)

    # Tính F1-score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    results = {
        'test_loss': float(loss),
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1_score': float(f1_score)
    }

    print(f"\n   📊 KẾT QUẢ:")
    print(f"   {'='*40}")
    print(f"   Loss:      {loss:.4f}")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1_score:.4f}")
    print(f"   {'='*40}")

    # Predictions cho confusion matrix
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   📋 CONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"                 Benign  Attack")
    print(f"   Actual Benign  {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"   Actual Attack  {cm[1,0]:>6}  {cm[1,1]:>6}")

    print(f"\n   📋 CLASSIFICATION REPORT:")
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Attack'])
    print(report)

    # Lưu classification report dạng dictionary
    report_dict = classification_report(y_test, y_pred, target_names=['Benign', 'Attack'], output_dict=True)

    # Thêm confusion matrix và các metrics khác vào results
    results['confusion_matrix'] = cm.tolist()
    results['classification_report'] = report_dict

    # Thêm các metrics chi tiết cho từng class
    results['benign_precision'] = float(report_dict['Benign']['precision'])
    results['benign_recall'] = float(report_dict['Benign']['recall'])
    results['benign_f1'] = float(report_dict['Benign']['f1-score'])
    results['attack_precision'] = float(report_dict['Attack']['precision'])
    results['attack_recall'] = float(report_dict['Attack']['recall'])
    results['attack_f1'] = float(report_dict['Attack']['f1-score'])

    # Tính thêm một số metrics bổ sung
    tn, fp, fn, tp = cm.ravel()
    results['true_negative'] = int(tn)
    results['false_positive'] = int(fp)
    results['false_negative'] = int(fn)
    results['true_positive'] = int(tp)
    results['specificity'] = float(tn / (tn + fp + 1e-7))  # True Negative Rate
    results['false_positive_rate'] = float(fp / (fp + tn + 1e-7))
    results['false_negative_rate'] = float(fn / (fn + tp + 1e-7))

    return results, y_pred, y_pred_prob


def save_model_and_results(model, history, results, training_time, model_dir, y_pred=None, y_pred_prob=None):
    """
    Lưu model và kết quả training

    Args:
        model: Trained model
        history: Training history
        results: Evaluation results
        training_time: Thời gian training (seconds)
        model_dir: Đường dẫn lưu
        y_pred: Predictions (optional)
        y_pred_prob: Prediction probabilities (optional)
    """
    print("\n" + "="*80)
    print("💾 ĐANG LƯU MODEL VÀ KẾT QUẢ...")
    print("="*80)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Lưu model cuối cùng
    final_model_path = model_dir / 'final_model.keras'
    model.save(final_model_path)
    print(f"   ✅ Final model: {final_model_path}")

    # Lưu model weights
    weights_path = model_dir / 'model_weights.weights.h5'
    model.save_weights(weights_path)
    print(f"   ✅ Model weights: {weights_path}")

    # Lưu training history
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(model_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"   ✅ Training history: training_history.json")

    # Lưu kết quả đánh giá với thông tin bổ sung
    results['training_time_seconds'] = float(training_time)
    results['training_time_minutes'] = float(training_time / 60)
    results['epochs_trained'] = int(len(history.history['loss']))

    # Thêm best validation metrics
    results['best_val_loss'] = float(min(history.history['val_loss']))
    results['best_val_accuracy'] = float(max(history.history['val_accuracy']))
    results['best_val_precision'] = float(max(history.history['val_precision']))
    results['best_val_recall'] = float(max(history.history['val_recall']))

    # Tính best val F1-score
    val_precisions = history.history['val_precision']
    val_recalls = history.history['val_recall']
    val_f1_scores = [2 * (p * r) / (p + r + 1e-7) for p, r in zip(val_precisions, val_recalls)]
    results['best_val_f1_score'] = float(max(val_f1_scores))

    # Epoch nào đạt best val_loss
    results['best_val_loss_epoch'] = int(np.argmin(history.history['val_loss']) + 1)
    results['best_val_accuracy_epoch'] = int(np.argmax(history.history['val_accuracy']) + 1)

    with open(model_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"   ✅ Evaluation results: evaluation_results.json")

    # Lưu predictions nếu có
    if y_pred is not None:
        np.save(model_dir / 'y_pred.npy', y_pred)
        print(f"   ✅ Predictions: y_pred.npy")

    if y_pred_prob is not None:
        np.save(model_dir / 'y_pred_prob.npy', y_pred_prob)
        print(f"   ✅ Prediction probabilities: y_pred_prob.npy")

    # Lưu cấu hình training
    config = {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'dropout_rate': DROPOUT_RATE,
        'patience': PATIENCE,
        'random_seed': RANDOM_SEED,
        'tensorflow_version': tf.__version__,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(model_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print(f"   ✅ Training config: training_config.json")

    print(f"\n📁 Tất cả file được lưu tại: {model_dir}")


def plot_training_history(history, model_dir):
    """
    Vẽ biểu đồ training history

    Args:
        history: Training history
        model_dir: Đường dẫn lưu hình
    """
    try:
        import matplotlib.pyplot as plt

        model_dir = Path(model_dir)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. Precision
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. Recall
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(model_dir / 'training_history.png', dpi=150)
        plt.close()
        print(f"   ✅ Training history plot: training_history.png")

    except ImportError:
        print("   ⚠️ matplotlib không có sẵn, bỏ qua việc vẽ biểu đồ")


def plot_confusion_matrix(cm, model_dir):
    """
    Vẽ confusion matrix dưới dạng heatmap

    Args:
        cm: Confusion matrix (numpy array hoặc list)
        model_dir: Đường dẫn lưu hình
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        model_dir = Path(model_dir)

        # Convert to numpy array if needed
        if isinstance(cm, list):
            cm = np.array(cm)

        # Vẽ heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig(model_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Confusion matrix plot: confusion_matrix.png")

        # Vẽ normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'],
                   cbar_kws={'label': 'Percentage'})
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig(model_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Normalized confusion matrix plot: confusion_matrix_normalized.png")

    except ImportError as e:
        print(f"   ⚠️ matplotlib/seaborn không có sẵn, bỏ qua việc vẽ confusion matrix: {e}")


def main():
    """Hàm chính để train model"""

    print("\n" + "="*80)
    print("🧠 HUẤN LUYỆN MÔ HÌNH CNN - PHÁT HIỆN LƯU LƯỢNG MẠNG BẤT THƯỜNG")
    print("   Binary Classification: Benign vs Attack")
    print("="*80)

    # Bước 1: Load dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_training_data(TRAINING_DATA_DIR)

    # Bước 2: Xây dựng mô hình
    input_shape = (X_train.shape[1], X_train.shape[2])  # (n_features, 1)
    model = build_cnn_model(input_shape)

    # Bước 3: Tạo callbacks
    callbacks = create_callbacks(MODEL_DIR, LOG_DIR)

    # Bước 4: Huấn luyện
    history, training_time = train_model(
        model, X_train, y_train, X_val, y_val, class_weights, callbacks
    )

    # Bước 5: Đánh giá
    results, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)

    # Bước 6: Lưu model và kết quả
    save_model_and_results(model, history, results, training_time, MODEL_DIR, y_pred, y_pred_prob)

    # Bước 7: Vẽ biểu đồ
    plot_training_history(history, MODEL_DIR)

    # Bước 8: Vẽ confusion matrix
    if 'confusion_matrix' in results:
        plot_confusion_matrix(results['confusion_matrix'], MODEL_DIR)

    print("\n" + "="*80)
    print("✅ HOÀN THÀNH HUẤN LUYỆN!")
    print(f"   Test Accuracy:  {results['test_accuracy']*100:.2f}%")
    print(f"   Test Precision: {results['test_precision']*100:.2f}%")
    print(f"   Test Recall:    {results['test_recall']*100:.2f}%")
    print(f"   Test F1-Score:  {results['test_f1_score']*100:.2f}%")
    print("="*80)

    return model, history, results


if __name__ == "__main__":
    model, history, results = main()


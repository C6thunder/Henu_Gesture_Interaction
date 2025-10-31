# train_keypoint_classifier.py
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================== 配置参数 ==================
CSV_PATH = "CSV/main_csv/main_data/keypoint.csv"          # CSV 数据路径
MODEL_SAVE_PATH = "test_keypoint_data/run/model/keypoint_classifier.tflite"  # TFLite 保存路径
BEST_MODEL_PATH = "test_keypoint_data/run/model/best_model.h5"      # Keras 保存路径
CURVE_SAVE_PATH = "test_keypoint_data/run/img/training_curve.png" # 训练曲线保存路径
EPOCHS = 500
BATCH_SIZE = 16
INPUT_DIM = 42                  # 21 个关键点 * 2
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3              # 防止过拟合
EARLY_STOPPING_MIN_DELTA = 1e-4 # loss 减少幅度低于此值视为无提升
EARLY_STOPPING_PATIENCE = 20    # loss 无改善连续多少轮停止
ENABLE_EARLY_STOPPING = True    # 是否启用早停
# ============================================

# ---------------- 数据读取 ----------------
def load_data(csv_path):
    X, y = [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != INPUT_DIM + 1:
                continue
            y.append(int(float(row[0])))
            X.append([float(x) for x in row[1:]])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

# ---------------- 模型定义 ----------------
def build_model(input_dim, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------- 自定义回调 ----------------
class TFLiteCheckpoint(keras.callbacks.Callback):
    """在保存最佳 Keras 模型的同时导出 TFLite 模型"""
    def __init__(self, h5_path, tflite_path, monitor='val_loss'):
        super().__init__()
        self.h5_path = h5_path
        self.tflite_path = tflite_path
        self.monitor = monitor
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best:
            self.best = current
            # 保存 Keras 模型
            self.model.save(self.h5_path)
            print(f"✅ Epoch {epoch+1}: val_loss improved, saved Keras model to {self.h5_path}")
            # 导出 TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            with open(self.tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"✅ 同步保存 TFLite 模型至 {self.tflite_path}")

# ---------------- 绘制训练曲线 ----------------
def plot_training_curve(history, save_path):
    plt.figure(figsize=(10,5))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📈 训练曲线已保存至 {save_path}")

# ---------------- 主函数 ----------------
def main():
    X, y = load_data(CSV_PATH)
    if len(X) == 0:
        print("❌ 数据为空，请检查 CSV")
        return

    NUM_CLASSES = len(np.unique(y))
    print(f"检测到手势类别数量: {NUM_CLASSES}")
    print("样本数量:", len(X))
    print("X shape:", X.shape, "y shape:", y.shape)

    model = build_model(INPUT_DIM, NUM_CLASSES)
    model.summary()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y
    )

    # ---------------- 回调 ----------------
    callbacks = [TFLiteCheckpoint(BEST_MODEL_PATH, MODEL_SAVE_PATH, monitor='val_loss')]
    if ENABLE_EARLY_STOPPING:
        earlystop_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=EARLY_STOPPING_MIN_DELTA,
            patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1
        )
        callbacks.append(earlystop_cb)

    print("\n训练开始，按 Ctrl+C 可中断，之后可选择继续或结束\n")

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("\n⚠️ 训练中断。")
        cont = input("输入 'y' 继续训练，其他键结束: ").strip().lower()
        if cont == 'y':
            remaining_epochs = EPOCHS - len(model.history.history['loss'])
            print(f"继续训练 {remaining_epochs} 轮...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=remaining_epochs,
                batch_size=BATCH_SIZE,
                shuffle=True,
                callbacks=callbacks
            )
        else:
            print("训练结束，使用当前模型或最佳模型。")

    # ---------------- 导出最终 TFLite ----------------
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    if os.path.exists(BEST_MODEL_PATH):
        model = keras.models.load_model(BEST_MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(MODEL_SAVE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"\n✅ TFLite 模型已保存至 {MODEL_SAVE_PATH}")
    print(f"📂 Keras 最佳模型保存在 {BEST_MODEL_PATH}")

    # ---------------- 绘制训练曲线 ----------------
    plot_training_curve(history, CURVE_SAVE_PATH)

if __name__ == "__main__":
    main()

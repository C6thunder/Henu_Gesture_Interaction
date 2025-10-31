# train_keypoint_classifier.py
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback  # 进度条

# ================== 配置参数 ==================
CSV_PATH = "CSV/main_csv/main_data/keypoint.csv"          # CSV 数据路径
BEST_MODEL_SAVE_PATH = "test_keypoint_data/run/model/best_keypoint_classifier.tflite"  # TFLite 保存路径
LAST_MODEL_SAVE_PATH = "test_keypoint_data/run/model/last_keypoint_classifier.tflite"  # TFLite 保存路径

BEST_MODEL_PATH = "test_keypoint_data/run/model/best_model.h5"      # Keras 保存路径
LAST_MODEL_PATH = "test_keypoint_data/run/model/last_model.h5"      # 最后一轮模型
CURVE_SAVE_PATH = "test_keypoint_data/run/img/training_curve.png"   # 训练曲线保存路径

EPOCHS = 50
BATCH_SIZE = 16
INPUT_DIM = 42                  # 21 个关键点 * 2
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
SAVE_EVERY_N_EPOCHS = 10        # 每隔多少轮保存一次模型

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
class PeriodicCheckpoint(keras.callbacks.Callback):
    """每隔 N 轮保存最佳模型和最后模型"""
    def __init__(self, save_every_n_epochs, best_h5_path, last_h5_path, monitor='val_loss'):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.best_h5_path = best_h5_path
        self.last_h5_path = last_h5_path
        self.monitor = monitor
        self.best_val = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get(self.monitor)
        if current_val is None:
            return

        # 更新最佳模型
        if current_val < self.best_val:
            self.best_val = current_val
            os.makedirs(os.path.dirname(self.best_h5_path), exist_ok=True)
            self.model.save(self.best_h5_path)
            print(f"\n✅ Epoch {epoch+1}: val_loss 改进，保存最佳模型至 {self.best_h5_path}")

        # 每隔固定轮保存 last 模型
        if (epoch + 1) % self.save_every_n_epochs == 0:
            os.makedirs(os.path.dirname(self.last_h5_path), exist_ok=True)
            self.model.save(self.last_h5_path)
            print(f"💾 Epoch {epoch+1}: 每 {self.save_every_n_epochs} 轮保存 last 模型至 {self.last_h5_path}")

# ---------------- h5 转 tflite ----------------
def convert_h5_to_tflite(h5_path, tflite_path):
    if not os.path.exists(h5_path):
        print(f"❌ 模型文件 {h5_path} 不存在")
        return
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ 已将 {h5_path} 转换为 TFLite 并保存至 {tflite_path}")

# ---------------- 绘制训练曲线 ----------------
def plot_training_curve(history, save_path):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
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

    callbacks = [PeriodicCheckpoint(
        save_every_n_epochs=SAVE_EVERY_N_EPOCHS,
        best_h5_path=BEST_MODEL_PATH,
        last_h5_path=LAST_MODEL_PATH
    ), TqdmCallback(verbose=1)]

    print("\n训练开始，按 Ctrl+C 可中断\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=0  # tqdm 显示进度条
    )

    # ---------------- 最后一轮保存 last 模型 ----------------
    os.makedirs(os.path.dirname(LAST_MODEL_PATH), exist_ok=True)
    model.save(LAST_MODEL_PATH)
    print(f"\n📂 最后一轮模型已保存至 {LAST_MODEL_PATH}")

    # ---------------- 转 TFLite ----------------
    convert_h5_to_tflite(BEST_MODEL_PATH, BEST_MODEL_SAVE_PATH)
    convert_h5_to_tflite(LAST_MODEL_PATH, LAST_MODEL_SAVE_PATH)

    # ---------------- 绘制训练曲线 ----------------
    plot_training_curve(history, CURVE_SAVE_PATH)

if __name__ == "__main__":
    main()

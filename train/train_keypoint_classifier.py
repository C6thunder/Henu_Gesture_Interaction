import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback

# ================== 配置参数 ==================
CSV_PATH = "CSV/main_csv/main_data/keypoint.csv"
BEST_MODEL_SAVE_PATH = "test_keypoint_data/run/model/best_keypoint_classifier.tflite"
LAST_MODEL_SAVE_PATH = "test_keypoint_data/run/model/last_keypoint_classifier.tflite"
BEST_MODEL_PATH = "test_keypoint_data/run/model/best_model.h5"
LAST_MODEL_PATH = "test_keypoint_data/run/model/last_model.h5"
CURVE_SAVE_PATH = "test_keypoint_data/run/img/training_curve.png"
SAMPLE_IMG_DIR = "test_keypoint_data/run/img/samples"   # 保存样本图目录

EPOCHS = 50
BATCH_SIZE = 16
INPUT_DIM = 42  # 21 点 * 2
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
SAVE_EVERY_N_EPOCHS = 10
# ============================================

# ---------------- 手部骨架连接定义 ----------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (9, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),# 小指
    (0, 17)                                # 手掌外框
]

# ---------------- 可视化函数 ----------------
def plot_hand_skeleton(points, save_path, label=None):
    """绘制单张手势关键点骨架图"""
    plt.figure(figsize=(3,3))
    x = points[::2]
    y = -points[1::2]
    plt.scatter(x, y, color='red', s=10)
    for (i, j) in HAND_CONNECTIONS:
        plt.plot([x[i], x[j]], [y[i], y[j]], color='blue', linewidth=1)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    plt.gca().add_patch(plt.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        fill=False, edgecolor='green', linewidth=1.5
    ))
    if label is not None:
        plt.title(f"Label: {label}")
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_hand_grid_with_boxes(X_samples, y_samples, save_path, cols=3):
    """绘制多个手势拼接图（类似YOLO样式）"""
    num_samples = len(X_samples)
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()

    for idx, (points, label) in enumerate(zip(X_samples, y_samples)):
        ax = axes[idx]
        x = points[::2]
        y = -points[1::2]
        ax.scatter(x, y, color='red', s=10)
        for (i, j) in HAND_CONNECTIONS:
            ax.plot([x[i], x[j]], [y[i], y[j]], color='blue', linewidth=1)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        ax.add_patch(plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor='green', linewidth=1.5
        ))
        ax.text(xmin, ymax + 0.02, f"Label: {label}", color='black', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.axis('off')

    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"🖼️ 拼接手势图已保存至 {save_path}")

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
    return np.array(X, np.float32), np.array(y, np.int32)

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
    """每隔 N 轮保存最佳模型和可视化结果"""
    def __init__(self, save_every_n_epochs, best_h5_path, last_h5_path, monitor, X_vis, y_vis):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.best_h5_path = best_h5_path
        self.last_h5_path = last_h5_path
        self.monitor = monitor
        self.best_val = np.Inf
        self.X_vis = X_vis
        self.y_vis = y_vis

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get(self.monitor)
        if current_val is None:
            return

        # 保存最佳模型
        if current_val < self.best_val:
            self.best_val = current_val
            os.makedirs(os.path.dirname(self.best_h5_path), exist_ok=True)
            self.model.save(self.best_h5_path)
            print(f"\n✅ Epoch {epoch+1}: val_loss 改进，保存最佳模型至 {self.best_h5_path}")

        # 每隔固定轮保存 last 模型与图像
        if (epoch + 1) % self.save_every_n_epochs == 0:
            os.makedirs(os.path.dirname(self.last_h5_path), exist_ok=True)
            self.model.save(self.last_h5_path)
            print(f"💾 Epoch {epoch+1}: 保存 last 模型至 {self.last_h5_path}")

            # 拼接图
            grid_path = os.path.join(SAMPLE_IMG_DIR, f"epoch_{epoch+1}_grid.png")
            plot_hand_grid_with_boxes(self.X_vis, self.y_vis, grid_path)

            # 单张图
            for i in range(min(3, len(self.X_vis))):
                single_path = os.path.join(SAMPLE_IMG_DIR, f"epoch_{epoch+1}_sample_{i}.png")
                plot_hand_skeleton(self.X_vis[i], single_path, label=self.y_vis[i])
            print(f"🖼️ 已保存 epoch_{epoch+1} 拼接图与单张骨架图")

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
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title('Accuracy')
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
    print(f"检测到手势类别数量: {NUM_CLASSES}, 样本数: {len(X)}")

    model = build_model(INPUT_DIM, NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y
    )

    # 随机选3个样本可视化
    vis_indices = np.random.choice(len(X_train), 3, replace=False)
    X_vis, y_vis = X_train[vis_indices], y_train[vis_indices]

    callbacks = [
        PeriodicCheckpoint(SAVE_EVERY_N_EPOCHS, BEST_MODEL_PATH, LAST_MODEL_PATH,
                           'val_loss', X_vis, y_vis),
        TqdmCallback(verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=0
    )

    model.save(LAST_MODEL_PATH)
    convert_h5_to_tflite(BEST_MODEL_PATH, BEST_MODEL_SAVE_PATH)
    convert_h5_to_tflite(LAST_MODEL_PATH, LAST_MODEL_SAVE_PATH)
    plot_training_curve(history, CURVE_SAVE_PATH)

if __name__ == "__main__":
    main()

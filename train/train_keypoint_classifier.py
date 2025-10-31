# train_keypoint_classifier.py
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
# import config
# ---------------- 配置 ----------------
CSV_PATH = "model/keypoint_classifier/keypoint.csv"          # CSV 数据路径
MODEL_SAVE_PATH = "keypoint_data/keypoint_classifier.tflite"  # 保存目录 + 文件名
EPOCHS = 500
BATCH_SIZE = 16
NUM_CLASSES = 4                        # 手势类别数
INPUT_DIM = 42                          # 21 个关键点 * 2

# ---------------- 数据读取 ----------------
# 修改 load_data 函数
def load_data(csv_path):
    X, y = [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != INPUT_DIM + 1:
                continue
            # 先 float 再 int
            y.append(int(float(row[0])))
            X.append([float(x) for x in row[1:]])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


# ---------------- 模型定义 ----------------
def build_model(input_dim, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------- 主函数 ----------------
def main():
    X, y = load_data(CSV_PATH)
    print("样本数量:", len(X))
    print("X shape:", X.shape, "y shape:", y.shape)
    if len(X) == 0:
        print("❌ 数据为空，请检查 CSV")
        return

    model = build_model(INPUT_DIM, NUM_CLASSES)
    model.summary()

    # 拆分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 确保保存目录存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 导出 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(MODEL_SAVE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite 模型已保存至 {MODEL_SAVE_PATH}")

# if __name__ == "__main__":
#     main()

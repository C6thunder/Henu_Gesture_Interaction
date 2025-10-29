#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

# -----------------------------
# 1. ONNX 模型加载
# -----------------------------
ONNX_MODEL_PATH = "/home/thunder/work/Prj/hand-gesture-recognition-mediapipe/hand_gesture/model/keypoint_classifier/keypoint_classifier.onnx"
sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

# -----------------------------
# 2. 类别标签
# -----------------------------
GESTURE_LABELS = ["Open", "Close", "Pointer", "OK", "Other1", "Other2"]  # 按 CSV 标签顺序

# -----------------------------
# 3. 关键点预处理
# -----------------------------
def preprocess_keypoints(hand_landmarks, img_width, img_height):
    """
    将 MediaPipe hand_landmarks 转换为归一化关键点向量
    """
    keypoints = []
    for lm in hand_landmarks.landmark:
        x = lm.x * img_width
        y = lm.y * img_height
        x = img_width - x  # 翻转水平
        keypoints.append(x)
        keypoints.append(y)

    keypoints = np.array(keypoints).reshape(1, -1)
    keypoints -= np.mean(keypoints)
    norm = np.linalg.norm(keypoints)
    if norm > 0:
        keypoints /= norm
    return keypoints.astype(np.float32)

# -----------------------------
# 4. 手势预测
# -----------------------------
def predict_gesture(keypoints):
    pred = sess.run(None, {input_name: keypoints})[0]  # shape (1, num_classes)
    pred_id = int(np.argmax(pred))
    pred_label = GESTURE_LABELS[pred_id]
    return pred_label, pred.flatten()

# -----------------------------
# 5. MediaPipe Hands 初始化
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# 6. 摄像头读取循环
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_height, img_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = preprocess_keypoints(hand_landmarks, img_width, img_height)
        gesture_label, pred_probs = predict_gesture(keypoints)

        # 输出到终端
        print(f"手势: {gesture_label}, 概率: {pred_probs}")

        # 绘制关键点
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture ONNX", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        break

cap.release()
cv2.destroyAllWindows()

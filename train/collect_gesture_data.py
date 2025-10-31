# collect_keypoints.py
import csv
import os
import cv2
import mediapipe as mp
import numpy as np

# ---------------- 配置 ----------------
DATA_DIR = "keypoint_data"
CSV_PATH = os.path.join(DATA_DIR, "keypoint_data.csv")
GESTURE_CLASSES = {
    0: "fist",
    1: "palm",
    2: "thumbs_up",
    3: "ok"
}
SAMPLES_PER_CLASS = 100  # 每个手势采集样本数

# ---------------- Mediapipe 初始化 ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# ---------------- CSV 初始化 ----------------
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        # 写表头可选，这里不写
        pass

# ---------------- 主函数 ----------------
def collect_keypoints():
    cap = cv2.VideoCapture(0)
    print("按 'q' 退出采集")
    for class_id, class_name in GESTURE_CLASSES.items():
        print(f"\n准备采集手势 '{class_name}' 共 {SAMPLES_PER_CLASS} 张")
        count = 0
        while count < SAMPLES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                # 绘制关键点
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 提取 21 个关键点 xy
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y])

                landmark_array = np.array([class_id] + landmark_list, dtype=np.float32)

                # 保存到 CSV
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(landmark_array)

                count += 1
                cv2.putText(frame, f"{class_name} {count}/{SAMPLES_PER_CLASS}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Keypoint Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
    cap.release()
    cv2.destroyAllWindows()
    print("采集完成！CSV 已保存:", CSV_PATH)

if __name__ == "__main__":
    collect_keypoints()

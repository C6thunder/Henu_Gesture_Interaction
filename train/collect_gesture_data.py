# collect_keypoints.py
import csv
import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm  # 用于进度条

# ================== 配置参数 ==================
DATA_DIR = "keypoint_data"          # CSV 保存目录
CSV_PATH = os.path.join(DATA_DIR, "keypoint_data.csv")
LABEL_PATH = os.path.join(DATA_DIR, "keypoint_label.csv")  # 手势标签 CSV
SAMPLES_PER_CLASS = 100             # 每个手势采集样本数
MIN_DETECTION_CONFIDENCE = 0.7      # Mediapipe 检测阈值
MIN_TRACKING_CONFIDENCE = 0.7       # Mediapipe 跟踪阈值
MAX_NUM_HANDS = 1                    # 最大手数
# ============================================

# ---------------- Mediapipe 初始化 ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)
mp_drawing = mp.solutions.drawing_utils

# ---------------- CSV 初始化 ----------------
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        pass

# ---------------- 主函数 ----------------
def collect_keypoints():
    # ---------------- 人工输入手势 ----------------
    GESTURE_CLASSES = {}
    num_classes = int(input("请输入要采集的手势数量: "))
    for i in range(num_classes):
        class_id = input(f"请输入第 {i+1} 个手势编号: ").strip()
        class_name = input(f"请输入第 {i+1} 个手势名称: ").strip()
        try:
            class_id = int(class_id)
        except ValueError:
            pass
        GESTURE_CLASSES[class_id] = class_name

    # 写入标签 CSV，每行一个手势名称（按 class_id 排序）
    sorted_classes = [v for k, v in sorted(GESTURE_CLASSES.items(), key=lambda x: x[0])]
    with open(LABEL_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        for label in sorted_classes:
            writer.writerow([label])
    print(f"手势标签已写入 {LABEL_PATH}")

    cap = cv2.VideoCapture(0)
    print("按 'q' 退出采集")

    for class_id, class_name in GESTURE_CLASSES.items():
        while True:  # 循环界面：可以重新采集
            print(f"\n准备采集手势 '{class_name}' 共 {SAMPLES_PER_CLASS} 张")
            count = 0
            pbar = tqdm(total=SAMPLES_PER_CLASS, desc=f"{class_name}", ncols=80)

            while count < SAMPLES_PER_CLASS:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmark_list = []
                    for lm in hand_landmarks.landmark:
                        landmark_list.extend([lm.x, lm.y])

                    landmark_array = np.array([class_id] + landmark_list, dtype=np.float32)

                    with open(CSV_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(landmark_array)

                    count += 1
                    pbar.update(1)

                    cv2.putText(frame, f"{count}/{SAMPLES_PER_CLASS}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Keypoint Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    pbar.close()
                    print("已退出采集")
                    return

            pbar.close()

            print(f"\n手势 '{class_name}' 采集完成！")
            action = input("输入 'r' 重新采集，回车继续下一个手势，'q' 退出: ").strip().lower()
            if action == 'r':
                print("重新采集...")
                continue
            elif action == 'q':
                cap.release()
                cv2.destroyAllWindows()
                print("已退出采集")
                return
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("所有手势采集完成！CSV 已保存:", CSV_PATH)


if __name__ == "__main__":
    collect_keypoints()

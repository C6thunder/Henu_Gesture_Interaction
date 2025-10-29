#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import subprocess
import time
from collections import defaultdict, deque, Counter

import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
import pyautogui
from pynput.mouse import Controller as MouseController, Button

# ---------------- HandGestureRecognition ----------------
class HandGestureRecognition:
    def __init__(self, device=0, width=960, height=540,
                 static_image_mode=False, min_det_conf=0.7, min_trk_conf=0.5):
        self.cap = cv.VideoCapture(device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
                # 在 __init__ 中增加点击冷却
        self.click_last_time = 0
        self.click_cooldown = 0.5  # 0.5秒一次点击，防止频繁触发
        # 在 __init__ 中增加鼠标平滑缓冲
        self.cursor_history = deque(maxlen=5)  # 保存最近 5 帧的坐标
        # 在 __init__ 中增加视频窗口尺寸属性
        self.cap_width = 960
        self.cap_height = 540
                # 在 __init__ 中增加灵敏度参数
        self.cursor_sensitivity = 1.5  # 1.0 为原始速度，>1加快，<1减慢
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_trk_conf
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_labels = [row[0] for row in csv.reader(f)]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_labels = [row[0] for row in csv.reader(f)]

        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.point_history = defaultdict(lambda: deque(maxlen=16))
        self.finger_gesture_history = defaultdict(lambda: deque(maxlen=16))

        self.mode = 0
        self.number = -1

        self.csv_buffer = []
        self.csv_buffer_size = 20

    def select_mode(self, key):
        if 48 <= key <= 57:
            self.number = key - 48
        if key == 110:  # n
            self.mode = 0
        if key == 107:  # k
            self.mode = 1
        if key == 104:  # h
            self.mode = 2

    @staticmethod
    def calc_bounding_rect(image, landmarks):
        image_w, image_h = image.shape[1], image.shape[0]
        coords = np.array([[min(int(l.x * image_w), image_w - 1),
                            min(int(l.y * image_h), image_h - 1)]
                           for l in landmarks.landmark])
        x, y, w, h = cv.boundingRect(coords)
        return [x, y, x + w, y + h]

    @staticmethod
    def calc_landmark_list(image, landmarks):
        image_w, image_h = image.shape[1], image.shape[0]
        return [[min(int(l.x * image_w), image_w - 1),
                 min(int(l.y * image_h), image_h - 1)]
                for l in landmarks.landmark]

    @staticmethod
    def pre_process_landmark(landmarks):
        landmarks = np.array(landmarks, dtype=np.float32)
        base = landmarks[0]
        landmarks -= base
        max_value = np.max(np.abs(landmarks)) or 1
        return (landmarks / max_value).flatten().tolist()

    @staticmethod
    def pre_process_point_history(point_history, image_shape):
        if not point_history:
            return [0.0] * 32
        ph = np.array(point_history, dtype=np.float32)
        base = ph[0]
        ph -= base
        ph[:, 0] /= image_shape[1]
        ph[:, 1] /= image_shape[0]
        return ph.flatten().tolist()

    def logging_csv(self, landmark_list, point_history_list):
        if self.mode == 1 and 0 <= self.number <= 9:
            self.csv_buffer.append(['keypoint', self.number, *landmark_list])
        elif self.mode == 2 and 0 <= self.number <= 9:
            self.csv_buffer.append(['pointhistory', self.number, *point_history_list])

        if len(self.csv_buffer) >= self.csv_buffer_size:
            for entry in self.csv_buffer:
                path = 'model/keypoint_classifier/keypoint.csv' if entry[0]=='keypoint' else 'model/point_history_classifier/point_history.csv'
                with open(path, 'a', newline='') as f:
                    csv.writer(f).writerow(entry[1:])
            self.csv_buffer.clear()

    @staticmethod
    def draw_hand_info(image, landmark_point, brect=None, handedness=None,
                       hand_sign_text="", finger_gesture_text="", point_history=None,
                       fps=None, mode=None, number=None, use_brect=True):
        if landmark_point:
            finger_bones = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],
                            [0,13,14,15,16],[0,17,18,19,20]]
            for finger in finger_bones:
                for i in range(len(finger)-1):
                    start, end = tuple(landmark_point[finger[i]]), tuple(landmark_point[finger[i+1]])
                    cv.line(image, start, end, (0,0,0), 6)
                    cv.line(image, start, end, (255,255,255), 2)
            keypoint_radius = {4:8, 8:8, 12:8, 16:8, 20:8}
            for idx, (x,y) in enumerate(landmark_point):
                r = keypoint_radius.get(idx, 5)
                cv.circle(image, (x,y), r, (255,255,255), -1)
                cv.circle(image, (x,y), r, (0,0,0), 1)

        if use_brect and brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0,0,0), 1)
            if handedness:
                cv.rectangle(image, (brect[0], brect[1]-22), (brect[2], brect[1]), (0,0,0), -1)
                text = handedness.classification[0].label
                if hand_sign_text:
                    text += f":{hand_sign_text}"
                cv.putText(image, text, (brect[0]+5, brect[1]-4),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)

        if finger_gesture_text:
            cv.putText(image, f"Finger Gesture:{finger_gesture_text}", (10,60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
            cv.putText(image, f"Finger Gesture:{finger_gesture_text}", (10,60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)

        if point_history:
            for idx, p in enumerate(point_history):
                if p[0] != 0 and p[1] != 0:
                    cv.circle(image, (p[0],p[1]), 1+idx//2, (152,251,152), 2)

        if fps is not None:
            cv.putText(image, f"FPS:{fps}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
            cv.putText(image, f"FPS:{fps}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)

        if mode and 1 <= mode <= 2:
            modes = ['Logging Key Point','Logging Point History']
            cv.putText(image, f"MODE:{modes[mode-1]}", (10,90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1, cv.LINE_AA)
            if number is not None and 0 <= number <= 9:
                cv.putText(image, f"NUM:{number}", (10,110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1, cv.LINE_AA)
        return image


# ---------------- HandGestureRecognitionWithCommand Optimized ----------------
class HandGestureRecognitionWithCommand(HandGestureRecognition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_enabled = False
        self.last_command_per_hand = defaultdict(lambda: (None, 0))
        self.command_cooldown = 1.0
        self.gesture_history = defaultdict(lambda: deque(maxlen=5))
        self.ok_last_time = 0
        self.ok_cooldown = 1.0

        self.mouse = MouseController()
        self.screen_w, self.screen_h = pyautogui.size()
        self.cursor_history = deque(maxlen=10)
        self.click_last_time = 0
        self.click_cooldown = 0.5
        self.cursor_sensitivity = 1.0
        self.chrome_open = False  # 避免重复开关 Chrome

        # 指数平滑系数
        self.smooth_alpha = 0.4

    def is_stable_gesture(self, hand_label, gesture, threshold=4):
        history = self.gesture_history[hand_label]
        history.append(gesture)
        if len(history) < threshold:
            return False
        most_common, count = Counter(history).most_common(1)[0]
        return most_common == gesture and count >= threshold // 2
    def move_cursor(self, landmark):
        """
        将手指坐标映射到屏幕坐标，并平滑移动鼠标。
        landmark: [x, y] 手指坐标（视频窗口坐标）
        """
        x, y = landmark

        # 映射到屏幕坐标
        screen_x = int((x / self.cap_width) * self.screen_w * self.cursor_sensitivity)
        screen_y = int((y / self.cap_height) * self.screen_h * self.cursor_sensitivity)

        # 限制在屏幕范围内（允许到边界）
        screen_x = max(0, min(self.screen_w - 1, screen_x))
        screen_y = max(0, min(self.screen_h - 1, screen_y))

        # 指数平滑
        if self.cursor_history:
            prev_x, prev_y = self.cursor_history[-1]
            avg_x = int(prev_x * (1 - self.smooth_alpha) + screen_x * self.smooth_alpha)
            avg_y = int(prev_y * (1 - self.smooth_alpha) + screen_y * self.smooth_alpha)
        else:
            avg_x, avg_y = screen_x, screen_y

        # 更新历史并移动鼠标
        self.cursor_history.append((avg_x, avg_y))
        self.mouse.position = (avg_x, avg_y)

    def execute_command(self, gesture):
        if gesture == "Open" and not self.chrome_open:
            subprocess.Popen(["google-chrome"])
            self.chrome_open = True
        elif gesture == "Close" and self.chrome_open:
            subprocess.Popen(["pkill", "chrome"])
            self.chrome_open = False

    def run(self):
        frame_count = 0
        while True:
            ret, image = self.cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)
            debug_image = image.copy()
            key = cv.waitKey(1)
            if key == 27:
                break
            self.select_mode(key)

            # 降低处理分辨率可选: 640x360
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                    pre_landmark = self.pre_process_landmark(landmark_list)
                    pre_point_history = self.pre_process_point_history(self.point_history[hand_label], image.shape)

                    self.logging_csv(pre_landmark, pre_point_history)

                    hand_sign_id = self.keypoint_classifier(pre_landmark)
                    hand_sign_name = self.keypoint_labels[hand_sign_id]

                    stable = self.is_stable_gesture(hand_label, hand_sign_name)

                    # OK 手势切换命令开关
                    if stable and hand_sign_name == "OK" and time.time() - self.ok_last_time > self.ok_cooldown:
                        self.command_enabled = not self.command_enabled
                        print(f"手势 OK 识别: {'允许执行命令' if self.command_enabled else '禁止命令'}")
                        self.ok_last_time = time.time()

                    # Open/Close 命令或 Pointer 鼠标控制
                    if stable:
                        if self.command_enabled and hand_sign_name in ["Open", "Close"]:
                            last_cmd, last_time = self.last_command_per_hand[hand_label]
                            if last_cmd != hand_sign_name or time.time() - last_time > self.command_cooldown:
                                self.execute_command(hand_sign_name)
                                self.last_command_per_hand[hand_label] = (hand_sign_name, time.time())
                        elif hand_sign_name == "Pointer":
                            self.move_cursor(landmark_list[8])
                        elif not self.command_enabled and hand_sign_name == "Close" and hand_label == "Left":
                            if time.time() - self.click_last_time > self.click_cooldown:
                                self.mouse.click(Button.left)
                                self.click_last_time = time.time()
                                print("模拟鼠标点击")

                    # 更新点历史
                    if hand_sign_name == "Pointer":
                        self.point_history[hand_label].append(landmark_list[8])
                    else:
                        self.point_history[hand_label].append([0, 0])

                    # 手指手势分类
                    finger_gesture_id = 0
                    if len(pre_point_history) == 32:
                        finger_gesture_id = self.point_history_classifier(pre_point_history)
                    self.finger_gesture_history[hand_label].append(finger_gesture_id)
                    most_common_fg = Counter(self.finger_gesture_history[hand_label]).most_common(1)[0][0]

                    debug_image = self.draw_hand_info(
                        debug_image,
                        landmark_point=landmark_list,
                        brect=brect,
                        handedness=handedness,
                        hand_sign_text=hand_sign_name,
                        finger_gesture_text=self.point_history_labels[most_common_fg],
                        point_history=self.point_history[hand_label],
                        fps=self.fps_calc.get() if frame_count % 5 == 0 else None,
                        mode=self.mode,
                        number=self.number
                    )

            cv.imshow('Hand Gesture Command', debug_image)
            frame_count += 1

        self.logging_csv([], [])
        self.cap.release()
        cv.destroyAllWindows()

# ---------------- Main ----------------
if __name__ == "__main__":
    app = HandGestureRecognitionWithCommand()
    # 可直接覆盖手势标签
    app.keypoint_labels = ["Open", "Close", "Pointer", "OK", "Other1", "Other2"]
    app.run()


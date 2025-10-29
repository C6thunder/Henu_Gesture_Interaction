#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hand_command.py
import time
import subprocess
from collections import defaultdict, deque, Counter
import pyautogui
from pynput.mouse import Controller as MouseController, Button
from hand_gesture import HandGestureRecognition
import cv2
import config  # 导入配置

class HandGestureRecognitionWithCommand(HandGestureRecognition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_enabled = False
        self.last_command_per_hand = defaultdict(lambda: (None, 0))
        self.command_cooldown = config.COMMAND_COOLDOWN
        self.gesture_history = defaultdict(lambda: deque(maxlen=config.GESTURE_HISTORY_LEN))
        self.ok_last_time = 0
        self.ok_cooldown = config.OK_COOLDOWN

        self.mouse = MouseController()
        self.screen_w, self.screen_h = pyautogui.size()
        self.cursor_history = deque(maxlen=config.CURSOR_HISTORY_LEN)
        self.click_last_time = 0
        self.click_cooldown = config.CLICK_COOLDOWN
        self.cursor_sensitivity = config.CURSOR_SENSITIVITY
        self.chrome_open = False
        self.smooth_alpha = config.SMOOTH_ALPHA

    def is_stable_gesture(self, hand_label, gesture):
        history = self.gesture_history[hand_label]
        history.append(gesture)
        if len(history) < config.GESTURE_STABLE_THRESHOLD:
            return False
        most_common, count = Counter(history).most_common(1)[0]
        return most_common == gesture and count >= config.GESTURE_STABLE_THRESHOLD // 2


    def move_cursor(self, landmark):
        x, y = landmark
        screen_x = int((x / self.cap_width) * self.screen_w * self.cursor_sensitivity)
        screen_y = int((y / self.cap_height) * self.screen_h * self.cursor_sensitivity)
        screen_x = max(0, min(self.screen_w - 1, screen_x))
        screen_y = max(0, min(self.screen_h - 1, screen_y))
        if self.cursor_history:
            prev_x, prev_y = self.cursor_history[-1]
            avg_x = int(prev_x * (1 - self.smooth_alpha) + screen_x * self.smooth_alpha)
            avg_y = int(prev_y * (1 - self.smooth_alpha) + screen_y * self.smooth_alpha)
        else:
            avg_x, avg_y = screen_x, screen_y
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
            image = cv2.flip(image, 1)
            debug_image = image.copy()
            key = cv2.waitKey(1)
            if key == 27:
                break
            self.select_mode(key)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

                    if stable and hand_sign_name == "OK" and time.time() - self.ok_last_time > self.ok_cooldown:
                        self.command_enabled = not self.command_enabled
                        print(f"手势 OK 识别: {'允许执行命令' if self.command_enabled else '禁止命令'}")
                        self.ok_last_time = time.time()

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

                    if hand_sign_name == "Pointer":
                        self.point_history[hand_label].append(landmark_list[8])
                    else:
                        self.point_history[hand_label].append([0, 0])

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

            cv2.imshow('Hand Gesture Command', debug_image)
            frame_count += 1

        self.logging_csv([], [])
        self.cap.release()
        cv2.destroyAllWindows()

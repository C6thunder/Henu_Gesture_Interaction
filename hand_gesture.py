import csv
from collections import defaultdict, deque
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
import config

class HandGestureRecognition:
    def __init__(self, device=config.CAP_DEVICE, width=config.CAP_WIDTH, height=config.CAP_HEIGHT,
                 static_image_mode=config.STATIC_IMAGE_MODE,
                 min_det_conf=config.MIN_DETECTION_CONF,
                 min_trk_conf=config.MIN_TRACKING_CONF):

        self.cap = cv.VideoCapture(device)
        self.cap_width = width
        self.cap_height = height
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        self.click_last_time = 0
        self.click_cooldown = config.CLICK_COOLDOWN
        self.cursor_history = deque(maxlen=config.CURSOR_HISTORY_LEN)
        self.cursor_sensitivity = config.CURSOR_SENSITIVITY

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_trk_conf
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        with open(config.KEYPOINT_CLASSIFIER_LABEL_PATH, encoding='utf-8-sig') as f:
            self.keypoint_labels = [row[0] for row in csv.reader(f)]
        with open(config.POINT_HISTORY_CLASSIFIER_LABEL_PATH, encoding='utf-8-sig') as f:
            self.point_history_labels = [row[0] for row in csv.reader(f)]

        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.point_history = defaultdict(lambda: deque(maxlen=config.POINT_HISTORY_LEN))
        self.finger_gesture_history = defaultdict(lambda: deque(maxlen=config.FINGER_GESTURE_HISTORY_LEN))
        self.mode = 0
        self.number = -1

        self.csv_buffer = []
        self.csv_buffer_size = config.CSV_BUFFER_SIZE

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
                path = config.KEYPOINT_CSV_PATH if entry[0]=='keypoint' else config.POINT_HISTORY_CSV_PATH
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
                    cv.line(image, start, end, config.DRAW_FINGER_LINE_COLOR, config.DRAW_FINGER_LINE_THICKNESS)
                    cv.line(image, start, end, config.DRAW_FINGER_LINE_OUTER_COLOR, 2)

            for idx, (x,y) in enumerate(landmark_point):
                r = config.DRAW_KEYPOINT_SPECIAL_RADIUS.get(idx, config.DRAW_KEYPOINT_RADIUS)
                cv.circle(image, (x,y), r, (255,255,255), -1)
                cv.circle(image, (x,y), r, (0,0,0), 1)

        if use_brect and brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), config.DRAW_BRECT_COLOR, 1)
            if handedness:
                cv.rectangle(image, (brect[0], brect[1]-22), (brect[2], brect[1]), config.DRAW_BRECT_COLOR, -1)
                text = handedness.classification[0].label
                if hand_sign_text:
                    text += f":{hand_sign_text}"
                cv.putText(image, text, (brect[0]+5, brect[1]-4),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)

        if finger_gesture_text:
            cv.putText(image, f"Finger Gesture:{finger_gesture_text}", config.FINGER_TEXT_POS,
                       cv.FONT_HERSHEY_SIMPLEX, config.FPS_FONT_SCALE, (0,0,0), config.FPS_THICKNESS_OUT, cv.LINE_AA)
            cv.putText(image, f"Finger Gesture:{finger_gesture_text}", config.FINGER_TEXT_POS,
                       cv.FONT_HERSHEY_SIMPLEX, config.FPS_FONT_SCALE, (255,255,255), config.FPS_THICKNESS_IN, cv.LINE_AA)

        if point_history:
            for idx, p in enumerate(point_history):
                if p[0] != 0 and p[1] != 0:
                    cv.circle(image, (p[0],p[1]), 1+idx//2, (152,251,152), 2)

        if fps is not None:
            cv.putText(image, f"FPS:{fps}", config.FPS_TEXT_POS,
                       cv.FONT_HERSHEY_SIMPLEX, config.FPS_FONT_SCALE, (0,0,0), config.FPS_THICKNESS_OUT, cv.LINE_AA)
            cv.putText(image, f"FPS:{fps}", config.FPS_TEXT_POS,
                       cv.FONT_HERSHEY_SIMPLEX, config.FPS_FONT_SCALE, (255,255,255), config.FPS_THICKNESS_IN, cv.LINE_AA)

        if mode and 1 <= mode <= 2:
            modes = ['Logging Key Point','Logging Point History']
            cv.putText(image, f"MODE:{modes[mode-1]}", config.MODE_TEXT_POS,
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1, cv.LINE_AA)
            if number is not None and 0 <= number <= 9:
                cv.putText(image, f"NUM:{number}", config.NUM_TEXT_POS,
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1, cv.LINE_AA)
        return image

import os
# --- 获取当前脚本所在目录 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 手势识别相关参数
COMMAND_COOLDOWN = 1.0        # 手势命令冷却时间
OK_COOLDOWN = 1.0             # OK手势切换命令模式冷却
CLICK_COOLDOWN = 0.5          # 鼠标点击冷却
GESTURE_STABLE_THRESHOLD = 4  # 手势稳定判定阈值

# 鼠标控制相关
CURSOR_SENSITIVITY = 1.0      # 鼠标灵敏度
SMOOTH_ALPHA = 0.4            # 鼠标平滑系数
CURSOR_HISTORY_LEN = 10       # 鼠标平滑历史长度

# 手势点历史长度
GESTURE_HISTORY_LEN = 5


# hand_gesture.py
# --- 摄像头配置 ---
CAP_DEVICE = 0
CAP_WIDTH = 960
CAP_HEIGHT = 540

# --- 鼠标控制 ---
CLICK_COOLDOWN = 0.5
CURSOR_HISTORY_LEN = 5
CURSOR_SENSITIVITY = 1.5

# --- Mediapipe Hands ---
STATIC_IMAGE_MODE = False
MIN_DETECTION_CONF = 0.7
MIN_TRACKING_CONF = 0.5

# --- 手势历史 ---
FINGER_GESTURE_HISTORY_LEN = 16    # 历史点要16个
POINT_HISTORY_LEN = 21   # 固定点要21个

# --- CSV 日志 ---
CSV_BUFFER_SIZE = 20

## --- 模型与数据路径（自动适配运行目录） ---
# csv标签
KEYPOINT_CLASSIFIER_LABEL_PATH = os.path.join(BASE_DIR, 'CSV/main_csv/main_lable/keypoint_classifier_label.csv')
POINT_HISTORY_CLASSIFIER_LABEL_PATH = os.path.join(BASE_DIR, 'CSV/main_csv/main_lable/point_history_classifier_label.csv')

# tflite模型
POINT_HISTORY_TFLITE_PATH = os.path.join(BASE_DIR, 'model/main_model/point_history_classifier.tflite')
# KEYPOINT_TFLITE_PATH = os.path.join(BASE_DIR, 'model/main_model/keypoint_classifier.tflite')

# csv数据
KEYPOINT_CSV_PATH = os.path.join(BASE_DIR, 'CSV/main_csv/main_data/keypoint.csv')
POINT_HISTORY_CSV_PATH = os.path.join(BASE_DIR, 'CSV/main_csv/main_data/point_history.csv')

# 用于测试
# KEYPOINT_CSV_PATH = os.path.join(BASE_DIR, 'model/keypoint_classifier/keypoint.csv')
KEYPOINT_TFLITE_PATH = os.path.join(BASE_DIR, 'test_keypoint_data/run/model/last_keypoint_classifier.tflite')


# --- 可视化参数 ---
DRAW_KEYPOINT_RADIUS = 5
DRAW_KEYPOINT_SPECIAL_RADIUS = {4:8, 8:8, 12:8, 16:8, 20:8}
DRAW_FINGER_LINE_COLOR = (0,0,0)
DRAW_FINGER_LINE_THICKNESS = 6
DRAW_FINGER_LINE_OUTER_COLOR = (255,255,255)
DRAW_BRECT_COLOR = (0,0,0)
FPS_TEXT_POS = (10,30)
FPS_FONT_SCALE = 1.0
FPS_THICKNESS_OUT = 4
FPS_THICKNESS_IN = 2
MODE_TEXT_POS = (10,90)
NUM_TEXT_POS = (10,110)
FINGER_TEXT_POS = (10,60)

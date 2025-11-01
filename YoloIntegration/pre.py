from ultralytics import YOLO

# ==========================
# 🚀 实时手势关键点检测（视频显示）
# ==========================

# 1. 加载模型（优先使用你训练好的模型）
model = YOLO("runs/pose/train_fast/weights/best.pt")  # 自训模型
# 如果想先测试官方模型，可以改成：
# model = YOLO("yolo11n-pose.pt")

# 2. 预测视频源（0 表示本机摄像头）
results = model.predict(
    source=0,       # 摄像头输入（也可换成 "video.mp4"）
    show=True,      # 实时显示检测画面
    conf=0.5,       # 置信度阈值
    stream=True,    # 实时流式预测
    device=0        # 使用 GPU
)

# 3. 获取关键点坐标（如需进一步处理）
for result in results:
    keypoints = result.keypoints
    if keypoints is not None:
        xy = keypoints.xy          # (x, y) 坐标
        xyn = keypoints.xyn        # 归一化坐标
        kpts = keypoints.data      # (x, y, visibility)
        # 你可以在这里添加自己的逻辑，如手势识别、轨迹绘制等

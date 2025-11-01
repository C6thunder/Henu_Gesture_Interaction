from ultralytics import YOLO

# ==========================
# 🚀 YOLO Pose 训练配置优化版
# ==========================

# 1. 加载预训练模型
model = YOLO("yolo11n-pose.pt")  # 预训练模型，建议继续 fine-tune

# 2. 开始训练
results = model.train(
    data="hand.yaml",            # 数据配置文件
    epochs=100,                  # 训练轮次
    imgsz=640,                   # 输入图像大小
    batch=32,                    # 批量大小（5060 显卡建议 32）
    workers=8,                   # dataloader 线程数
    device=0,                    # 指定使用 GPU:0
    cache=True,                  # 将数据缓存到内存，加快加载
    deterministic=False,         # 关闭确定性，启用 cuDNN 加速
    plots=True,                
    amp=True,                    # 启用混合精度（加速并节省显存）
    save=True,                   # 保存模型结果
    verbose=True,                # 打印详细日志
    name="train_fast",           # 保存路径 runs/pose/train_fast/
    project="runs/pose",         # 项目路径
)

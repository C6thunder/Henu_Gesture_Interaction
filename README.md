# 🤖 Henu_Gesture_Interaction

> 人机协同 · 手势控制 · 智能交互
> 基于 **MediaPipe + 深度学习分类模型** 的实时手势识别与系统控制项目

---

## 🧩 模块说明

| 模块                | 功能简介                             |
| ----------------- | -------------------------------- |
| `main.py`         | 🎬 主程序入口，捕捉摄像头视频并调用手势识别与控制模块     |
| `hand_gesture.py` | ✋ MediaPipe 检测手部关键点并调用分类模型进行手势识别 |
| `hand_command.py` | 🖱️ 根据识别到的手势执行系统指令（鼠标移动、点击等）     |
| `model/`          | 🧠 存放训练好的手势分类模型与标签文件             |
| `utils.py`        | ⚙️ FPS 计算、关键点归一化、图像绘制等辅助功能       |
| `config.py`       | 🧾 模型路径、摄像头索引、GPU 使用开关等全局配置      |

---

## 🧠 手势识别逻辑

系统基于 **MediaPipe Hands** 模块提取 21 个关键点坐标，
经过归一化处理后输入 **KeyPointClassifier** 模型，输出手势类别。

**流程图：**

```python
1️⃣ 捕获视频帧  
2️⃣ MediaPipe 检测手部关键点  
3️⃣ 关键点归一化处理  
4️⃣ KeyPointClassifier 推理手势类别  
5️⃣ HandCommand 执行系统操作
```

---

## ✋ 手势示例

<div align="center">

| 手势    | 示例图                                                     |
| ----- | ------------------------------------------------------- |
| Point | <img src="doc/Point.png" alt="手势演示:point" width="260"/> |
| Close | <img src="doc/Close.png" alt="手势演示:close" width="260"/> |
| Open  | <img src="doc/Open.png" alt="手势演示:open" width="260"/>   |
| OK    | <img src="doc/OK.png" alt="手势演示:OK" width="260"/>       |

</div>

---

## 📁 文件结构

<details>
<summary>点击展开查看</summary>

```text
Henu_Gesture_Interaction
├──  config.py
├──  CSV
│  ├──  main_csv
│  │  ├──  main_data
│  │  │  ├──  keypoint.csv
│  │  │  └──  point_history.csv
│  │  └──  main_lable
│  │     ├──  keypoint_classifier_label.csv
│  │     └──  point_history_classifier_label.csv
│  └──  other_csv
├──  doc
│  ├──  Close.png
│  ├──  OK.png
│  ├──  Open.png
│  ├──  Point.png
│  └──  preview.png
├──  hand_command.py
├──  hand_gesture.py
├──  keypoint_data
│  └──  keypoint_data.csv
├──  LICENSE
├──  main.py
├──  model
│  ├──  main_model
│  │  ├──  keypoint_classifier.tflite
│  │  └──  point_history_classifier.tflite
│  └──  other_model
│     └──  keypoint_classifier.tflite
├──  README.md
├──  requirements.txt
├──  train
│  ├──  __init__.py
│  ├──  collect_gesture_data.py
│  └──  train_keypoint_classifier.py
├──  ts.py
└──  utils
   ├──  __init__.py
   ├──  Classifier
   │  ├──  keypoint_classifier.py
   │  └──  point_history_classifier.py
   └──  cvfpscalc.py
```

</details>

---

## 🚀 扩展方向

* 🎥 **PPT / 浏览器 / 视频播放控制**
* 🤖 **机械臂动作模仿手势**
* 🗣️ **语音识别 + 手势识别 → 多模态人机交互**
* 🧩 **个性化手势训练 → 自定义控制方案**

---

## 🎬 手势演示视频

<div align="center">
  <a href="https://www.bilibili.com/video/BV1mmyaB1Eqo?t=0" target="_blank">
    <img src="doc/preview.png" alt="手势演示预览" width="720"/>
  </a>
  <br>
  <em>点击图片即可跳转至 Bilibili 视频演示</em>
</div>

---

## 🧑‍💻 开发环境

* Python ≥ 3.10
* OpenCV ≥ 4.8
* MediaPipe ≥ 0.10
* TensorFlow Lite / ONNX Runtime
* NumPy, PyAutoGUI, time, csv 等常用库

---

## 📜 License

本项目遵循 **MIT License** 开源协议，欢迎学习与二次开发。
如使用于论文、竞赛或展示，请注明出处：
**Henu_Gesture_Interaction 团队项目**

---

✨ *让计算机“看懂”手势，让交互更自然。*

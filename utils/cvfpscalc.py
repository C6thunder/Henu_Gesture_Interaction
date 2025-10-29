# utils/cvfpscalc.py
import time
from collections import deque

class CvFpsCalc:
    """
    计算 FPS 或每帧耗时的工具类
    - buffer_len: 用于平滑 FPS 的帧数
    """
    def __init__(self, buffer_len=30):
        self._prev_time = time.perf_counter()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self, return_fps=True):
        """
        获取当前 FPS 或每帧耗时（ms）
        :param return_fps: True 返回 FPS, False 返回每帧耗时(ms)
        :return: float
        """
        current_time = time.perf_counter()
        diff = (current_time - self._prev_time) * 1000  # 转为毫秒
        self._prev_time = current_time

        self._difftimes.append(diff)
        avg_time = sum(self._difftimes) / len(self._difftimes)

        return round(1000.0 / avg_time, 2) if return_fps else round(avg_time, 2)

    def reset(self):
        """重置计时器和缓冲区"""
        self._prev_time = time.perf_counter()
        self._difftimes.clear()

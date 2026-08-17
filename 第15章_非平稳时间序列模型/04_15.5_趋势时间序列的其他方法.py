# 04_15.5 趋势时间序列的其他方法

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def hp_filter(y, lam=1600.0):
    """H-P 滤波: 解 (I + lam D'D) tau = y, D 为二阶差分阵。"""
    T = len(y)
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1; D[i, i + 1] = -2; D[i, i + 2] = 1
    A = np.eye(T) + lam * D.T @ D
    return np.linalg.solve(A, y)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    t = np.arange(T)
    trend = 100 + 0.5 * t + 0.001 * t ** 2
    cycle = 3 * np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.5, T)
    y = trend + cycle

    tau = hp_filter(y, 1600.0)
    c = y - tau
    print("H-P 趋势 τ 末值:", round(tau[-1], 2), " (总体均值", round(trend[-1], 2), ")")
    print("循环 c 应含主导周期(标准差):", round(c.std(), 2))
    # c 应接近 cycle
    print("(τ+ c =y 分解正确:", round(np.max(np.abs(tau + c - y)), 10), ")")
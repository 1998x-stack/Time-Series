# 02_9.3 识别

"""
Lecture: /第9章 线性联立方程系统
Content: 02_9.3 识别
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    # 系统: 内生总数 G=2;总外生 K=2 (x1,x2)
    G, K = 2, 2
    # 方程1: 含内生 g1=1 (y2), 含外生 k1=1 (x1), 排除 x2
    g1, k1 = 1, 1
    excluded = K - k1             # 排除的外生=1
    need = g1 - 1                 # 需要 ≥ 1
    print(f"方程1: 被排除外生 = K-k1 = {K}-{k1} = {excluded}, 需要 g1-1 = {need}")
    if excluded > need:
        print("状态: 过度识别 (over-identified)")
    elif excluded == need:
        print("状态: 恰识别 (just-identified)")
    else:
        print("状态: 不足识别 (under-identified)")

    # 模拟恰识别, 2SLS 恢复 alpha
    rng = np.random.default_rng(2026)
    T = 400
    x1 = rng.normal(size=T); x2 = rng.normal(size=T)
    e2 = rng.normal(size=T); e1 = 0.6 * e2 + 0.8 * rng.normal(size=T)
    y2 = x1 + 0.7 * x2 + e2
    y1 = 0.8 * y2 + 0.5 * x1 + e1
    Z = np.column_stack([x1, x2])
    y2hat = Z @ np.linalg.lstsq(Z, y2, rcond=None)[0]
    a_tsls = np.linalg.lstsq(np.column_stack([y2hat, x1]), y1, rcond=None)[0][0]
    print("恰识别下 2SLS alpha =", round(a_tsls, 4), " (真 0.8)")
# 07_第9章习题

"""
Lecture: /第9章 线性联立方程系统
Content: 07_第9章习题
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def dgp(T, rng):
    x1 = rng.normal(size=T); x2 = rng.normal(size=T)
    e2 = rng.normal(size=T)
    e1 = 0.6 * e2 + 0.8 * rng.normal(size=T)
    y2 = x1 + 0.7 * x2 + e2
    y1 = 0.8 * y2 + 0.5 * x1 + e1
    return y1, y2, x1, x2


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 5000
    y1, y2, x1, x2 = dgp(T, rng)

    # 习题2: 两阶段
    Z = np.column_stack([x1, x2])
    first = np.linalg.lstsq(Z, y2, rcond=None)[0]
    y2hat = Z @ first
    second = np.linalg.lstsq(np.column_stack([y2hat, x1]), y1, rcond=None)[0]
    print(f"习题2(2SLS): 第一阶段 y2={np.round(first,3)}, 第二阶段 alpha={second[0]:.4f}")

    # 习题1: OLS 对照
    ols_beta = np.linalg.lstsq(np.column_stack([y2, x1]), y1, rcond=None)[0]
    print(f"习题1(OLS): alpha={ols_beta[0]:.4f} (有偏, 真0.8)")

    # 习题3: 阶条件 G=2 需 G-1=1 个工具
    print("习题3: 阶条件被排除变量≥ G-1; 本系统 G=2 → 至少 1 个工具(x2)")
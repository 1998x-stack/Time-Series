# 04_9.5 基于简化型的估计

"""
Lecture: /第9章 线性联立方程系统
Content: 04_9.5 基于简化型的估计
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 8000
    a, b1, b21, b22 = 0.8, 0.5, 1.0, 0.7
    x1 = rng.normal(size=T); x2 = rng.normal(size=T)
    e2 = rng.normal(size=T); e1 = 0.6 * e2 + 0.8 * rng.normal(size=T)
    y2 = b21 * x1 + b22 * x2 + e2
    y1 = a * y2 + b1 * x1 + e1

    W = np.column_stack([np.ones(T), x1, x2])
    # 简化型
    pi1 = np.linalg.lstsq(W, y1, rcond=None)[0]   # coef [c, x1, x2]
    pi2 = np.linalg.lstsq(W, y2, rcond=None)[0]
    alpha_ils = pi1[2] / pi2[2]
    # 2SLS 对照
    y2hat = W @ pi2
    alpha_tsls = np.linalg.lstsq(np.column_stack([y2hat, x1]), y1, rcond=None)[0][0]
    print("ILS alpha =", round(alpha_ils, 4), "  2SLS alpha =", round(alpha_tsls, 4),
          "  真值 =", a)
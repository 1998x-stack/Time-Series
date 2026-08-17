# 01_9.2 工具变量与两阶段最小二乘

"""
Lecture: /第9章 线性联立方程系统
Content: 01_9.2 工具变量与两阶段最小二乘法
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def dgp(T, rng, alpha=0.8, beta1=0.5, b21=1.0, b22=0.7, rho=0.6):
    x1 = rng.normal(size=T); x2 = rng.normal(size=T)
    e2 = rng.normal(size=T)
    e1 = rho * e2 + np.sqrt(1 - rho ** 2) * rng.normal(size=T)
    y2 = b21 * x1 + b22 * x2 + e2
    y1 = alpha * y2 + beta1 * x1 + e1
    return y1, y2, x1, x2


def tsls(y1, y2, x1, x2):
    """2SLS: 用 Z=(x1,x2) 工具内生 y2。"""
    Z = np.column_stack([x1, x2])
    y2hat = Z @ np.linalg.lstsq(Z, y2, rcond=None)[0]
    X2 = np.column_stack([y2hat, x1])
    return np.linalg.lstsq(X2, y1, rcond=None)[0][0]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    alpha = 0.8
    T = 400
    B = 3000
    ba_ols = np.zeros(B); ba_tsls = np.zeros(B)
    for b in range(B):
        y1, y2, x1, x2 = dgp(T, rng)
        ba_ols[b] = np.linalg.lstsq(np.column_stack([y2, x1]), y1, rcond=None)[0][0]
        ba_tsls[b] = tsls(y1, y2, x1, x2)

    print("OLS  的 E[α̂] =", round(ba_ols.mean(), 4), "(偏, 真 0.8)")
    print("2SLS 的 E[α̂] =", round(ba_tsls.mean(), 4), "(一致, 接近 0.8)")
    print("2SLS 无偏程度:", round(abs(ba_tsls.mean() - alpha), 4))
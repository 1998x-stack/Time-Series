# 05_9.6 联立方程偏差综述

"""
Lecture: /第9章 线性联立方程系统
Content: 05_9.6 联立方程偏差综述
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


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    alpha = 0.8
    T, B = 300, 3000
    ba_ols = np.zeros(B); ba_tsls = np.zeros(B)
    for b in range(B):
        y1, y2, x1, x2 = dgp(T, rng)
        ba_ols[b] = np.linalg.lstsq(np.column_stack([y2, x1]), y1, rcond=None)[0][0]
        Z = np.column_stack([x1, x2])
        y2hat = Z @ np.linalg.lstsq(Z, y2, rcond=None)[0]
        ba_tsls[b] = np.linalg.lstsq(np.column_stack([y2hat, x1]), y1, rcond=None)[0][0]

    print("方法           E[alpha_hat]  偏差")
    print(f"OLS            {ba_ols.mean():.4f}       {ba_ols.mean()-alpha:+.4f}")
    print(f"2SLS:          {ba_tsls.mean():.4f}       {ba_tsls.mean()-alpha:+.4f}")
    print(f"真值 alpha = {alpha}")
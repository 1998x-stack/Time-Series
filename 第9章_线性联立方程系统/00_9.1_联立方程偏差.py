# 00_9.1 联立方程偏差

"""
Lecture: /第9章 线性联立方程系统
Content: 00_9.1 联立方程偏差
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def dgp(T, rng, alpha=0.8, beta1=0.5, b21=1.0, b22=0.7, rho=0.6):
    """两方程系统: y1 = alpha*y2 + beta1*x1 + eps1; y2 内生。"""
    x1 = rng.normal(size=T)
    x2 = rng.normal(size=T)
    e2 = rng.normal(size=T)
    e1 = rho * e2 + np.sqrt(1 - rho ** 2) * rng.normal(size=T)
    y2 = b21 * x1 + b22 * x2 + e2
    y1 = alpha * y2 + beta1 * x1 + e1
    return y1, y2, x1, x2


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    alpha = 0.8
    T = 400
    B = 3000
    alphas = np.zeros(B)
    for b in range(B):
        y1, y2, x1, x2 = dgp(T, rng)
        X = np.column_stack([y2, x1])
        bet = np.linalg.lstsq(X, y1, rcond=None)[0]
        alphas[b] = bet[0]

    print("OLS 的 E[alpha_hat] =", round(alphas.mean(), 4),
          "  真值 alpha =", alpha)
    print("联立方程偏差 =", round(alphas.mean() - alpha, 4),
          " (非零且不随 T 消失)")
    print("OLS 一致? 否. 解法: IV/2SLS (9.2)")
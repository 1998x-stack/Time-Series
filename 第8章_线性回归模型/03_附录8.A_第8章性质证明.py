# 03_附录8.A 第8章性质证明

"""
Lecture: /第8章 线性回归模型
Content: 03_附录8.A 第8章性质证明
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T, k = 300, 3
    X = np.column_stack([np.ones(T), rng.normal(size=T), rng.normal(size=T)])
    true_beta = np.array([1.0, 2.0, -0.5])

    B = 3000
    betas = np.zeros((B, k))
    for b in range(B):
        y = X @ true_beta + rng.normal(size=T)
        betas[b] = np.linalg.lstsq(X, y, rcond=None)[0]

    print("MC E[beta_hat] =", np.round(betas.mean(0), 4),
          " (真 beta =", true_beta, ")")
    print("MC 无偏误差:", np.round(betas.mean(0) - true_beta, 5))

    # 单次残差正交性
    y0 = X @ true_beta + rng.normal(size=T)
    e = y0 - X @ np.linalg.lstsq(X, y0, rcond=None)[0]
    print("X'残差 =", np.round(X.T @ e, 8), " (≈0, 正交)")
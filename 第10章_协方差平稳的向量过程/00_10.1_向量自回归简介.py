# 00_10.1 向量自回归简介

"""
Lecture: /第10章 协方差平稳的向量过程
Content: 00_10.1 向量自回归简介
"""

import numpy as np


def simulate_var1(Phi, Omega, T, rng):
    """VAR(1): y_t = Phi y_{t-1} + eps_t。返回 (T, n) 序列。"""
    n = Phi.shape[0]
    L = np.linalg.cholesky(Omega)
    burn = 200
    total = T + burn
    y = np.zeros((total, n))
    for t in range(1, total):
        y[t] = Phi @ y[t - 1] + L @ rng.normal(size=n)
    return y[burn:]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi = np.array([[0.5, 0.1], [0.2, 0.4]])
    Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
    ev = np.linalg.eigvals(Phi)
    print("Phi 特征值:", np.round(ev, 4), " 模=", np.round(np.abs(ev), 4),
          " 平稳:", bool(np.max(np.abs(ev)) < 1))

    T = 4000
    Y = simulate_var1(Phi, Omega, T, rng)
    Y = Y - Y.mean(0)
    print("样本协方差 Σ = Cov(y):")
    print(np.round(np.cov(Y.T), 4))

    # 各分量自相关 (h=1)
    r1 = np.corrcoef(Y[1:, 0], Y[:-1, 0])[0, 1]
    print("y1 一阶自相关:", round(r1, 3), "  y2 一阶自相关:",
          round(np.corrcoef(Y[1:, 1], Y[:-1, 1])[0, 1], 3))
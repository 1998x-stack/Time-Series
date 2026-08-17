# 04_10.5 向量过程的样本均值

"""
Lecture: /第10章 协方差平稳的向量过程
Content: 04_10.5 向量过程的样本均值
"""

import numpy as np


def simulate_var1(Phi, Omega, T, rng):
    n = Phi.shape[0]
    L = np.linalg.cholesky(Omega)
    burn = 200; total = T + burn
    y = np.zeros((total, n))
    for t in range(1, total):
        y[t] = Phi @ y[t - 1] + L @ rng.normal(size=n)
    return y[burn:]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi = np.array([[0.5, 0.1], [0.2, 0.4]])
    Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
    n = 2
    T = 400
    B = 3000
    ybar = np.zeros((B, n))
    for b in range(B):
        Y = simulate_var1(Phi, Omega, T, rng)
        ybar[b] = Y.mean(0)

    # 经验: T * Var(ybar)
    Lambda_emp = T * np.cov(ybar.T)
    # 理论: 2π F(0) = (I-Phi)^{-1} Omega (I-Phi')^{-1}
    Lambda_th = np.linalg.inv(np.eye(n) - Phi) @ Omega @ np.linalg.inv(np.eye(n) - Phi.T)
    print("经验 T Var(ybar):\n", np.round(Lambda_emp, 3))
    print("理论 2πF(0):\n", np.round(Lambda_th, 3))
# 01_10.2 向量过程的自协方差与收敛性结论

"""
Lecture: /第10章 协方差平稳的向量过程
Content: 01_10.2 向量过程的自协方差与收敛性结论
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


def sample_gamma_pos(Y, h):
    """样本 Gamma(h) = (1/(T-h)) sum_t y_{t+h} y_t' (去均值后), h>=0。"""
    T, n = Y.shape
    G = np.zeros((n, n))
    for t in range(T - h):
        G += np.outer(Y[t + h], Y[t])
    return G / (T - h)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi = np.array([[0.5, 0.1], [0.2, 0.4]])
    Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
    T = 30000
    Y = simulate_var1(Phi, Omega, T, rng)
    Y = Y - Y.mean(0)

    G1 = sample_gamma_pos(Y, 1)
    Gm1 = sample_gamma_pos(Y, 1).T   # Gamma(-1) = Gamma(1)'
    print("Gamma(1)  =\n", np.round(G1, 4))
    print("Gamma(-1) 应=Gamma(1)':\n", np.round(Gm1, 4))
    print("Gamma(1) ≈ Gamma(1)'(对称逆推一致性) 差:",
          np.round(np.max(np.abs(G1 - G1.T)), 6))
    G0 = sample_gamma_pos(Y, 0)
    print("Gamma(0) 特征值(应>=0):", np.round(np.linalg.eigvalsh(G0), 4))
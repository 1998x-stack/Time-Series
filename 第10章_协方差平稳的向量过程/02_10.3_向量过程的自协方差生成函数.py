# 02_10.3 向量过程的自协方差生成函数

# 向量 AGF 与样本对照

import numpy as np


def agf_var1(Phi, Omega, z):
    n = Phi.shape[0]
    A = np.linalg.inv(np.eye(n) - Phi * z)
    B = np.linalg.inv(np.eye(n) - Phi.T / z)
    return A @ Omega @ B


def sample_gamma(Y, h):
    T, n = Y.shape
    G = np.zeros((n, n))
    for t in range(T - h):
        G += np.outer(Y[t + h], Y[t])
    return G / (T - h)


def sample_agf(Y, H, z):
    out = sample_gamma(Y, 0).copy()
    for h in range(1, H + 1):
        Gh = sample_gamma(Y, h)
        out += Gh * z ** h + Gh.T * z ** (-h)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi = np.array([[0.5, 0.1], [0.2, 0.4]])
    Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
    n = 2
    T = 200000
    L = np.linalg.cholesky(Omega)
    burn = 200; total = T + burn
    y = np.zeros((total, n))
    for t in range(1, total):
        y[t] = Phi @ y[t - 1] + L @ rng.normal(size=n)
    Y = y[burn:] - y[burn:].mean(0)

    z = 1.0
    G_an = agf_var1(Phi, Omega, z)
    G_emp = sample_agf(Y, 200, z)
    print("G(z=1) 解析:\n", np.round(G_an, 3))
    print("G(z=1) 样本:\n", np.round(G_emp, 3))
    print("二者最大差:", np.round(np.max(np.abs(G_an - G_emp)), 3))
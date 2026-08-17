# 07_附录11.A 第11章性质证明

import numpy as np


def lyapunov_var1(Phi, Omega, iters=2000):
    """迭代解李雅普诺夫: Gamma = Phi Gamma Phi' + Omega。"""
    G = np.zeros_like(Omega)
    for _ in range(iters):
        G = Phi @ G @ Phi.T + Omega
    return G


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi = np.array([[0.5, 0.2], [0.1, 0.3]])
    Omega = np.array([[1.2, 0.4], [0.4, 0.9]])
    n = 2

    G_th = lyapunov_var1(Phi, Omega)
    L = np.linalg.cholesky(Omega)
    T = 20000
    burn = 100; total = T + burn
    Y = np.zeros((total, n))
    for t in range(1, total):
        Y[t] = Phi @ Y[t - 1] + L @ rng.normal(size=n)
    Y = Y[burn:]
    G_sample = np.cov(Y.T, bias=True)

    print("李雅普诺夫 Γ(0):\n", np.round(G_th, 4))
    print("样本   Γ(0):\n", np.round(G_sample, 4))
    print("最大差:", np.round(np.max(np.abs(G_th - G_sample)), 4))
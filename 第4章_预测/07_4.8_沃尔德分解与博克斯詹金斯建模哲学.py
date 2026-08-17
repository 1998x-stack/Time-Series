# 07_4.8 沃尔德分解与博克斯詹金斯建模哲学

"""
Lecture: /第4章 预测
Content: 07_4.8 沃尔德分解与博克斯詹金斯建模哲学
"""

import numpy as np


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    n = len(x)
    xc = x - x.mean()
    return np.array([np.dot(xc[h:], xc[: n - h]) / np.dot(xc, xc)
                     for h in range(maxlag + 1)])


def wold_psi(phi, theta, J):
    """ARMA 的沃尔德(脉冲)系数 psi_j。"""
    p, q = len(phi), len(theta)
    th = np.concatenate([[1.0], theta])
    psi = np.zeros(J + 1)
    psi[0] = 1.0
    for k in range(1, J + 1):
        psi[k] = th[k] if k <= q else 0.0
        for i in range(1, min(p, k) + 1):
            psi[k] += phi[i - 1] * psi[k - i]
    return psi


def simulate_arma(phi, theta, sigma, n, rng):
    p, q = len(phi), len(theta)
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(total):
        val = eps[t]
        for i in range(p):
            if t - 1 - i >= 0:
                val += phi[i] * x[t - 1 - i]
        for j in range(q):
            if t - 1 - j >= 0:
                val += theta[j] * eps[t - 1 - j]
        x[t] = val
    return x[burn:], eps[burn:]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi = np.array([0.6])
    theta = np.array([0.4])
    sigma = 1.0
    n = 20000

    psi = wold_psi(phi, theta, 8)
    print("Wold 脉冲 psi_0..8:", np.round(psi, 4), " (几何衰减)")

    x, eps_true = simulate_arma(phi, theta, sigma, n, rng)
    # 反解新息: eps_t = x_t - phi x_{t-1} - theta eps_{t-1}
    rec = np.zeros(n)
    for t in range(n):
        rec[t] = x[t]
        if t - 1 >= 0:
            rec[t] -= phi[0] * x[t - 1]
            rec[t] -= theta[0] * rec[t - 1]
    rho = sample_acf(rec, 4)
    print("还原新息 ACF (h=1..4):", np.round(rho[1:], 4), " (≈0, 白)")
    print("还原新息方差:", round(rec.var(), 4), "≈ 1")
# 04_3.5 混合自回归移动平均过程

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 04_3.5 混合自回归移动平均过程
"""

import numpy as np


def simulate_arma(phi: np.ndarray, theta: np.ndarray, sigma: float,
                  n: int, rng) -> np.ndarray:
    """ARMA(p,q) 模拟 (预热后返回)。"""
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
    return x[burn:]


def impulse_response(phi: np.ndarray, theta: np.ndarray, J: int) -> np.ndarray:
    """脉冲响应 psi_0..J: psi_k = theta_k + sum_{i<=min(p,k)} phi_i psi_{k-i}。"""
    p, q = len(phi), len(theta)
    theta_full = np.concatenate([[1.0], theta])
    psi = np.zeros(J + 1)
    psi[0] = 1.0
    for k in range(1, J + 1):
        psi[k] = theta_full[k] if k <= q else 0.0
        for i in range(1, min(p, k) + 1):
            psi[k] += phi[i - 1] * psi[k - i]
    return psi


def theory_acf(phi: np.ndarray, theta: np.ndarray, sigma: float,
               maxlag: int) -> np.ndarray:
    """ARMA 理论自协方差 gamma(h)=sigma^2 * sum_j psi_j psi_{j+h} (截断)。"""
    J = maxlag + 500
    psi = impulse_response(phi, theta, J)
    g = np.zeros(maxlag + 1)
    for h in range(maxlag + 1):
        g[h] = sigma ** 2 * np.sum(psi[: J - h + 1] * psi[h:])
    return g


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    """样本自协方差 (T 归一化)。"""
    n = len(x)
    xc = x - x.mean()
    return np.array([
        np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)
    ])


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 10000
    sigma = 1.0
    phi = np.array([0.6])       # AR(1) 部分
    theta = np.array([0.4])     # MA(1) 部分

    x = simulate_arma(phi, theta, sigma, n, rng)
    ghat = sample_acf(x, 5)
    gth = theory_acf(phi, theta, sigma, 5)
    print("样本 gamma_hat:", np.round(ghat, 4))
    print("理论 gamma   :", np.round(gth, 4))

    psi = impulse_response(phi, theta, 3)
    print("脉冲响应 psi_0..3:", np.round(psi, 4),
          " (psi_1=phi+theta 应 =", round(phi[0] + theta[0], 4), ")")
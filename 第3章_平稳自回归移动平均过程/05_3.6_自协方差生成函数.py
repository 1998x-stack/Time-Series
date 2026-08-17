# 05_3.6 自协方差生成函数

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 05_3.6 自协方差生成函数
"""

import numpy as np


def impulse_response(phi: np.ndarray, theta: np.ndarray, J: int) -> np.ndarray:
    """psi_0..J: psi_k = theta_k + sum_{j<=min(p,k)} phi_j psi_{k-j}。"""
    p, q = len(phi), len(theta)
    th = np.concatenate([[1.0], theta])
    psi = np.zeros(J + 1)
    psi[0] = 1.0
    for k in range(1, J + 1):
        psi[k] = th[k] if k <= q else 0.0
        for i in range(1, min(p, k) + 1):
            psi[k] += phi[i - 1] * psi[k - i]
    return psi


def agf_theory(phi: np.ndarray, theta: np.ndarray, sigma: float,
               lam: np.ndarray, J: int) -> np.ndarray:
    """理论 AGF 在单位圆 z=e^{-iλ}: g=sigma^2 |Psi(z)|^2。"""
    psi = impulse_response(phi, theta, J)
    z = np.exp(-1j * lam)
    Psi = sum(psi[j] * z ** j for j in range(J + 1))
    return sigma ** 2 * np.abs(Psi) ** 2


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    """样本自协方差 gamma_hat(h), h=0..maxlag。"""
    n = len(x)
    xc = x - x.mean()
    return np.array([
        np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)
    ])


def agf_sample(x: np.ndarray, lam: np.ndarray, H: int) -> np.ndarray:
    """样本 AGF: sum_{h=-H}^{H} gamma_hat(h) z^h, z=e^{-iλ}。"""
    g = sample_acf(x, H)
    z = np.exp(-1j * lam)
    out = np.zeros(len(lam))
    for k, zk in enumerate(z):
        total = g[0]
        for h in range(1, H + 1):
            total += g[h] * (zk ** h + zk ** (-h))
        out[k] = np.real(total)
    return out


def simulate_arma(phi: np.ndarray, theta: np.ndarray, sigma: float,
                  n: int, rng) -> np.ndarray:
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


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 20000
    sigma = 1.0
    phi = np.array([0.6])
    theta = np.array([0.4])
    lam = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])

    theory = agf_theory(phi, theta, sigma, lam, J=400)
    x = simulate_arma(phi, theta, sigma, n, rng)
    sample = agf_sample(x, lam, H=60)

    print("λ:", np.round(lam, 3))
    print("AGF 理论谱:", np.round(theory, 3))
    print("样本 AGF  :", np.round(sample, 3))
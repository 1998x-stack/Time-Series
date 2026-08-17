# 03_3.4 自回归过程

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 03_3.4 自回归过程
"""

import numpy as np


def simulate_ar(phi: np.ndarray, sigma: float, n: int, rng) -> np.ndarray:
    """AR(p): x_t = sum phi_i x_{t-i} + eps_t (预热后返回)。"""
    p = len(phi)
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(total):
        val = eps[t]
        for i in range(p):
            if t - 1 - i >= 0:
                val += phi[i] * x[t - 1 - i]
        x[t] = val
    return x[burn:]


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    """样本自协方差 (T 归一化)。"""
    n = len(x)
    xc = x - x.mean()
    return np.array([
        np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)
    ])


def yule_walker_acf(phi: np.ndarray, sigma: float, maxlag: int) -> np.ndarray:
    """AR(p) 理论自协方差 gamma(0..maxlag)。

    对 h=0..p 解线性方程组, 再用 gamma(h)=sum phi_i gamma(h-i) 递推。
    """
    p = len(phi)
    A = np.zeros((p + 1, p + 1))
    b = np.zeros(p + 1)
    A[0, 0] = 1.0
    for i in range(p):
        A[0, i + 1] = -phi[i]
    b[0] = sigma ** 2
    for h in range(1, p + 1):
        A[h, h] = 1.0
        for i in range(p):
            A[h, abs(h - (i + 1))] -= phi[i]
    g = np.zeros(maxlag + 1)
    g[: p + 1] = np.linalg.solve(A, b)
    for h in range(p + 1, maxlag + 1):
        g[h] = sum(phi[i] * g[h - i - 1] for i in range(p))
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 8000
    sigma = 1.0

    # AR(1)
    phi1 = np.array([0.7])
    x1 = simulate_ar(phi1, sigma, n, rng)
    g1 = sample_acf(x1, 6)
    t1 = yule_walker_acf(phi1, sigma, 6)
    print("AR(1) 样本 gamma_hat:", np.round(g1, 4))
    print("AR(1) 理论 gamma   :", np.round(t1, 4), "(几何衰减, 不截零)")

    # AR(2)
    phi2 = np.array([0.5, -0.2])
    x2 = simulate_ar(phi2, sigma, n, rng)
    g2 = sample_acf(x2, 6)
    t2 = yule_walker_acf(phi2, sigma, 6)
    print("\nAR(2) 样本 gamma_hat:", np.round(g2[:5], 4))
    print("AR(2) 理论 gamma   :", np.round(t2[:5], 4))

    # 非平稳根: 方差随时间增长
    x3 = simulate_ar(np.array([1.03]), sigma, 2000, rng)
    var_first = x3[:400].var()
    var_last = x3[-400:].var()
    print(f"\nAR(1), phi=1.03: 前400方差={var_first:.2f} 后400方差={var_last:.2f} (方差增长=>非平稳)")
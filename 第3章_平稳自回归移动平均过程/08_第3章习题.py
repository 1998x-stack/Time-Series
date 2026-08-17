# 08_第3章习题

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 08_第3章习题
"""

import numpy as np


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    """样本自协方差 (T 归一化)。"""
    n = len(x)
    xc = x - x.mean()
    return np.array([
        np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)
    ])


def simulate_ar1(phi: float, sigma: float, n: int, rng) -> np.ndarray:
    burn = 200
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    prev = 0.0
    for t in range(total):
        prev = phi * prev + eps[t]
        x[t] = prev
    return x[burn:]


def simulate_ma1(theta: float, sigma: float, n: int, rng) -> np.ndarray:
    burn = 200
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    pe = 0.0
    for t in range(total):
        x[t] = eps[t] + theta * pe
        pe = eps[t]
    return x[burn:]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 20000

    # 习题1: AR(1) phi=0.7
    phi = 0.7
    g0 = 1.0 / (1 - phi ** 2)
    theory = np.array([phi ** h * g0 for h in range(4)])
    x1 = simulate_ar1(phi, 1.0, n, rng)
    est = sample_acf(x1, 3)
    print("习题1 AR(1)φ=0.7: 理论 γ=", np.round(theory, 3), " 样本=", np.round(est, 3))

    # 习题2: MA(1) theta=0.6
    theta = 0.6
    th = np.array([1.0 + theta ** 2, theta, 0.0, 0.0])
    x2 = simulate_ma1(theta, 1.0, n, rng)
    est2 = sample_acf(x2, 3)
    print("习题2 MA(1)θ=0.6: 理论 γ=", np.round(th, 3),
          " 样本=", np.round(est2, 3), " 可逆:", abs(theta) < 1)

    # 习题3: AR(1) phi=1.2 非平稳
    print("习题3: φ=1.2 相伴特征值=1.2>1 => 非平稳")
    # 习题4: MA(2) theta=(0.8,-0.4) 可逆性: 特征多项式 1+0.8z-0.4z^2=0 之根|z|>1
    roots = np.roots([-0.4, 0.8, 1.0])
    print("习题4 MA(2) 根=", np.round(roots, 3),
          "  |根|=", np.round(np.abs(roots), 3), " => 可逆=",
          bool(np.all(np.abs(roots) > 1)))
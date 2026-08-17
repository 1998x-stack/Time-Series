# 02_3.3 移动平均过程

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 02_3.3 移动平均过程
"""

import numpy as np


def simulate_ma(theta: np.ndarray, sigma: float, n: int, rng) -> np.ndarray:
    """MA(q): x_t = eps_t + theta1 eps_{t-1} + ... (零初值, 热后丢弃预热)。"""
    q = len(theta)
    burn = 200
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(total):
        val = eps[t]
        for j in range(1, q + 1):
            if t - j >= 0:
                val += theta[j - 1] * eps[t - j]
        x[t] = val
    return x[burn:]


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    """样本自协方差 (T 归一化)。"""
    n = len(x)
    xc = x - x.mean()
    return np.array([
        np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)
    ])


def theory_acf(theta: np.ndarray, sigma: float, maxlag: int) -> np.ndarray:
    """MA(q) 理论自协方差 gamma(h)。"""
    q = len(theta)
    coef = np.concatenate([[1.0], theta])       # [1, theta1, ..., thetaq]
    g = np.zeros(maxlag + 1)
    for h in range(maxlag + 1):
        g[h] = sigma ** 2 * sum(coef[j] * coef[j + h]
                                for j in range(0, q + 1 - h))
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 8000
    sigma = 1.0
    theta = np.array([0.6, -0.4])

    x = simulate_ma(theta, sigma, n, rng)
    ghat = sample_acf(x, 5)
    gth = theory_acf(theta, sigma, 5)
    print("样本 gamma_hat:", np.round(ghat, 4))
    print("理论 gamma :", np.round(gth, 4))
    print("gamma(h>q=2) 应约 0:", np.round(ghat[3:], 4))

    # 任意系数下的 MA 依旧平稳 (自协方差有限且收敛)
    x2 = simulate_ma(np.array([2.5, -3.0]), sigma, n, rng)
    print("系数超界 MA 方差 =", round(x2.std() ** 2, 3), "(仍平稳)")
# 01_3.2 白噪声

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 01_3.2 白噪声
"""

import numpy as np


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    """样本自协方差 gamma_hat(h), h=0..maxlag (T 归一化)。"""
    n = len(x)
    xc = x - x.mean()
    return np.array([
        np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)
    ])


def periodogram(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """周期图功率谱 P(lambda) = |sum_t x_t e^{-i lambda t}|^2 / T。"""
    t = np.arange(len(x))
    P = np.zeros(len(freqs))
    for k, lam in enumerate(freqs):
        P[k] = np.abs(np.sum(x * np.exp(-1j * lam * t))) ** 2 / len(x)
    return P


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 8000
    sigma = 1.0

    eps = rng.normal(0.0, sigma, n)

    # 样本自协方差
    g = sample_acf(eps, 3)
    print("样本均值 =", round(eps.mean(), 4))
    print("样本自协方差 gamma_hat(0..3) =", np.round(g, 4), " (理论 [1,0,0,0])")
    print("gamma_hat(0) ≈ sigma^2:", round(g[0], 4), "≈", sigma ** 2)

    # 周期图功率谱在若干频率, 理论 f = sigma^2/(2pi)
    freqs = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
    P = periodogram(eps, freqs)
    theory = sigma ** 2 / (2 * np.pi)
    print("理论谱密度 f = sigma^2/(2pi) =", round(theory, 4))
    print("周期图样本:", np.round(P, 4), " (围绕理论值波动)")
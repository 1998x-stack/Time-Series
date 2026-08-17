# 00_3.1 期望、平稳性和遍历性

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 00_3.1 期望、平稳性和遍历性
"""

import numpy as np


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    """样本自协方差 gamma_hat(h), h=0..maxlag (使用总体 T 归一)。"""
    n = len(x)
    xc = x - x.mean()
    out = np.zeros(maxlag + 1)
    for h in range(maxlag + 1):
        out[h] = np.dot(xc[h:], xc[: n - h]) / n
    return out


def ar1_simulate(phi: float, sigma: float, n: int, rng) -> np.ndarray:
    """平稳 AR(1): x_t = phi x_{t-1} + eps_t (从平稳区预热)。"""
    x = np.zeros(n)
    prev = 0.0
    for t in range(n):
        prev = phi * prev + rng.normal(0.0, sigma)
        x[t] = prev
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 4000
    sig = 1.0

    # 1) 白噪声 (平稳遍历): 均值≈0, acf≈[1,0,0,...]
    wn = rng.normal(0.0, sig, n)
    g = sample_acf(wn, 3)
    print("白噪声: 样本均值 =", round(wn.mean(), 4), "  样本自协方差 =", np.round(g, 4))
    print("  理论 γ(0)=1, γ(h≠0)=0")

    # 2) 平稳 AR(1): 时间平均随 T 收敛于总体均值 0
    phi = 0.7
    x = ar1_simulate(phi, sig, n, rng)
    print("\nAR(1) 累积均值随 T 收敛(遍历性):")
    for T in (100, 500, 2000, 4000):
        print(f"  T={T:5d}  mean={x[:T].mean():+.4f}")

    # 3) 随机游走 (非平稳): 时间平均不收敛
    rw = np.zeros(n)
    prev = 0.0
    for t in range(n):
        prev = prev + rng.normal(0.0, sig)
        rw[t] = prev
    print("\n随机游走累积均值(非平稳, 不收敛):")
    for T in (100, 500, 2000, 4000):
        print(f"  T={T:5d}  cummean={rw[:T].mean():+.4f}")
# 02_6.3 总体谱估计

"""
Lecture: /第6章 谱分析
Content: 02_6.3 总体谱估计
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ar1_spectrum(phi, sigma, lam):
    return (sigma ** 2 / (2 * np.pi)) / np.abs(1 - phi * np.exp(-1j * lam)) ** 2


def sample_acf(x, maxlag):
    n = len(x)
    xc = x - x.mean()
    return np.array([np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)])


def smoothed_spectrum(g, M, lam):
    """Blackman-Tukey: fhat = (1/2π) Σ_{h=-M}^{M} (1-|h|/(M+1)) γ(h) e^{-iλh}。"""
    out = np.zeros(len(lam))
    for k, lk in enumerate(lam):
        val = g[0]
        for h in range(1, M + 1):
            wh = 1 - abs(h) / (M + 1)          # Bartlett 窗
            val += wh * g[h] * (np.exp(-1j * lk * h) + np.exp(1j * lk * h))
        out[k] = np.real(val) / (2 * np.pi)
    return np.real(out)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi, sigma = 0.7, 1.0
    T = 8000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.normal(0.0, sigma)

    lam = np.array([0.2, 0.6, 1.0, 1.4, 2.0, 2.6])
    M = 200
    g = sample_acf(x, M)
    fhat = smoothed_spectrum(g, M, lam)
    ftrue = ar1_spectrum(phi, sigma, lam)
    print("λ:", np.round(lam, 2))
    print("平滑谱 f̂:", np.round(fhat, 4))
    print("理论谱 f :", np.round(ftrue, 4))
    print("相对误差   :", np.round(np.abs(fhat - ftrue) / ftrue, 3))

    # 绘图: 光谱估计算与真谱
    grid = np.linspace(0.05, np.pi, 300)
    plt.figure(figsize=(6, 4))
    plt.plot(grid, smoothed_spectrum(g, M, grid), label="smoothed f-hat")
    plt.plot(grid, ar1_spectrum(phi, sigma, grid), "--", label="true f")
    plt.xlabel("lambda"); plt.ylabel("spectral density"); plt.legend()
    plt.savefig("plots/6.3_spectral_estimate.png")
    print("绘保存: plots/6.3_spectral_estimate.png")
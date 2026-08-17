# 03_6.4 谱分析的应用

"""
Lecture: /第6章 谱分析
Content: 03_6.4 谱分析的应用
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_acf(x, maxlag):
    n = len(x)
    xc = x - x.mean()
    return np.array([np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)])


def smoothed_spectrum(g, M, lam):
    out = np.zeros(len(lam))
    for k, lk in enumerate(lam):
        val = g[0]
        for h in range(1, M + 1):
            wh = 1 - abs(h) / (M + 1)
            val += wh * g[h] * (np.exp(-1j * lk * h) + np.exp(1j * lk * h))
        out[k] = np.real(val) / (2 * np.pi)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 4000
    t = np.arange(1, T + 1)
    # 主周期 40, 幅度 1; 次周期 10, 幅度 0.4; 噪声
    lam_main = 2 * np.pi / 40
    lam_sec = 2 * np.pi / 10
    y = np.sin(lam_main * t) + 0.4 * np.sin(lam_sec * t) + rng.normal(0.0, 0.5, T)

    # 平滑谱找峰
    M = 300
    grid = np.linspace(0.01, np.pi, 1000)
    g = sample_acf(y, M)
    spec = smoothed_spectrum(g, M, grid)
    k = np.argmax(spec)
    peak_lam = grid[k]
    period = 2 * np.pi / peak_lam
    print(f"谱峰频率 λ* = {peak_lam:.4f},  换算周期 = {period:.1f} 期 (主周期真值 40)")
    print(f"次峰频率 λ = {2*np.pi/10:.4f} (10 期) 也应在谱中")

    plt.figure(figsize=(6, 4))
    plt.plot(grid, spec, label="smoothed spectrum")
    plt.axvline(peak_lam, color="r", ls="--")
    plt.xlabel("lambda"); plt.ylabel("spectral density"); plt.legend()
    plt.savefig("plots/6.4_cycle_detection.png")
    print("绘图保存: plots/6.4_cycle_detection.png")
# 01_6.2 样本谱

"""
Lecture: /第6章 谱分析
Content: 01_6.2 样本谱
"""

import numpy as np


def ar1_spectrum(phi, sigma, lam):
    return (sigma ** 2 / (2 * np.pi)) / np.abs(1 - phi * np.exp(-1j * lam)) ** 2


def periodogram(x):
    """完整周期图在 DFT 频率上: I = |FFT|^2 / T。"""
    T = len(x)
    X = np.fft.fft(x)
    return np.abs(X) ** 2 / T


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi, sigma = 0.7, 1.0
    T = 4000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.normal(0.0, sigma)

    I = periodogram(x)
    # 取非负频率: lambda_k = 2pi k / T
    freqs = 2 * np.pi * np.arange(T) / T
    half = T // 2
    lam_band = freqs[:half]
    I_band = I[:half]

    # 频带平均 vs 理论谱
    bands = [(0.0, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, np.pi)]
    print("频带平均周期图 vs 理论谱:")
    for lo, hi in bands:
        mask = (lam_band >= lo) & (lam_band < hi)
        if mask.sum() == 0:
            continue
        avg = I_band[mask].mean()
        flam = ar1_spectrum(phi, sigma, (lo + hi) / 2)
        print(f"  [{lo:.2f},{hi:.2f}]  平均周期图={avg:.4f}  理论f={flam:.4f}")

    # 单点波动(不一致性): 取同一频带内两个点
    m = (lam_band > 0.5) & (lam_band < 0.6)
    pts = I_band[m]
    print("同频带内单点周期图波动 (不一致):", np.round(pts[:4], 3))
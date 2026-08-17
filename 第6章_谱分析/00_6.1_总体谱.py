# 00_6.1 总体谱

"""
Lecture: /第6章 谱分析
Content: 00_6.1 总体谱
"""

import numpy as np


def ar1_spectrum(phi, sigma, lam):
    """AR(1) 谱密度 f=lambda = sigma^2/(2pi) / |1-phi e^{-ilam}|^2。"""
    num = sigma ** 2 / (2 * np.pi)
    den = np.abs(1 - phi * np.exp(-1j * lam)) ** 2
    return num / den


def ar1_acf(phi, sigma, hmax):
    h = np.arange(hmax + 1)
    return sigma ** 2 * phi ** h / (1 - phi ** 2)


if __name__ == "__main__":
    phi, sigma = 0.7, 1.0
    lam = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    f = ar1_spectrum(phi, sigma, lam)
    print("λ:", np.round(lam, 3))
    print("AR1 谱 f(λ):", np.round(f, 3), " (低频高、高频低)")

    # 面积 ≈ 方差 (梯形或求和)
    grid = np.linspace(-np.pi, np.pi, 4001)
    fg = ar1_spectrum(phi, sigma, grid)
    area = np.trapz(fg, grid)
    var = ar1_acf(phi, sigma, 0)[0]
    print("∫f dλ =", round(area, 4), "  Var(x) = γ(0) =", round(var, 4),
          " (应相等)")
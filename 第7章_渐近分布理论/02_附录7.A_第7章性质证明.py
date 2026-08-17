# 02_附录7.A 第7章性质证明

"""
Lecture: /第7章 渐近分布理论
Content: 02_附录7.A 第7章性质证明
"""

import numpy as np


def ar1_acf(phi, sigma, H):
    h = np.arange(H + 1)
    return sigma ** 2 * phi ** h / (1 - phi ** 2)


def ar1_spectrum0(phi, sigma):
    return sigma ** 2 / (2 * np.pi * (1 - phi) ** 2)


if __name__ == "__main__":
    phi, sigma = 0.7, 1.0
    H = 200
    g = ar1_acf(phi, sigma, H)

    # (a) 时域截断和
    lam_td = g[0] + 2 * g[1:].sum()
    # (b) 零频谱 * 2pi
    lam_f0 = 2 * np.pi * ar1_spectrum0(phi, sigma)
    # (c) 闭式
    lam_ab = sigma ** 2 / (1 - phi) ** 2

    print("(a) 求和 Σγ(h)      =", round(lam_td, 4))
    print("(b) 2π·f(0)         =", round(lam_f0, 4))
    print("(c) σ²/(1-φ)²      =", round(lam_ab, 4))
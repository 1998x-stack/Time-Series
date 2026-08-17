# Durbin-Levinson算法

"""
Lecture: /第1章 差分方程
Content: Durbin-Levinson算法
"""

import numpy as np


def durbin_levinson(gamma: np.ndarray) -> tuple:
    """Durbin-Levinson 递推: 由自协方差 gamma[0..p] 求 AR(p) 系数与预报误差方差。

    Args:
        gamma: 长度 p+1, gamma[0] 为方差, gamma[k] 为滞后 k 的自协方差。

    Returns:
        (phi, sigma): phi 为长 p 的 AR 系数; sigma 为噪声方差(标量)。
    """
    p = len(gamma) - 1
    phi = np.zeros((p + 1, p + 1))   # phi[k, i], i=1..k
    sigma = np.zeros(p + 1)
    sigma[0] = gamma[0]
    for k in range(1, p + 1):
        numer = gamma[k]
        for j in range(1, k):
            numer -= phi[k - 1, j] * gamma[k - j]
        phi[k, k] = numer / sigma[k - 1]
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        sigma[k] = sigma[k - 1] * (1 - phi[k, k] ** 2)
    return phi[p, 1 : p + 1], sigma[p]


def main():
    rng = np.random.default_rng(7)
    true_phi = np.array([0.5, -0.2])
    T = 40000
    y = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(2, T):
        y[t] = true_phi[0] * y[t - 1] + true_phi[1] * y[t - 2] + e[t]
    y = y - y.mean()
    gamma = np.array([
        np.dot(y, y) / T,
        np.dot(y[1:], y[:-1]) / T,
        np.dot(y[2:], y[:-2]) / T,
    ])
    est, sig = durbin_levinson(gamma)
    print(f"估计系数: {np.round(est, 4)}   真值: {true_phi}")
    print(f"噪声方差估计: {sig:.4f}   理论: 1.0")
    print("最大系数误差:", np.abs(est - true_phi).max())


if __name__ == "__main__":
    main()
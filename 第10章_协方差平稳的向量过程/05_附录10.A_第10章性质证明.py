# 05_附录10.A 第10章性质证明

"""
Lecture: /第10章 协方差平稳的向量过程
Content: 05_附录10.A 第10章性质证明
"""

import numpy as np


def spectral_var1(Phi, Omega, lam):
    n = Phi.shape[0]
    ze = np.exp(-1j * lam)
    A = np.linalg.inv(np.eye(n) - Phi * ze)
    B = np.linalg.inv(np.eye(n) - Phi.T / ze)
    return (A @ Omega @ B) / (2 * np.pi)


if __name__ == "__main__":
    Phi = np.array([[0.5, 0.1], [0.2, 0.4]])
    Omega = np.array([[1.0, 0.3], [0.3, 1.0]])
    n = 2

    # (i) Gamma(0) = (I-Phi)^{-1} Omega (I-Phi')^{-1} 半正定
    G0 = np.linalg.inv(np.eye(n) - Phi) @ Omega @ np.linalg.inv(np.eye(n) - Phi.T)
    print("Gamma(0) 特征值:", np.round(np.linalg.eigvalsh(G0), 4), " (应≥0)")

    # (ii) 积分谱 = Gamma(0)
    grid = np.linspace(-np.pi, np.pi, 4001)
    F00 = np.array([spectral_var1(Phi, Omega, x)[0, 0] for x in grid])
    F11 = np.array([spectral_var1(Phi, Omega, x)[1, 1] for x in grid])
    F01 = np.array([spectral_var1(Phi, Omega, x)[0, 1] for x in grid])
    I00 = np.trapezoid(F00, grid); I11 = np.trapezoid(F11, grid)
    I01 = np.trapezoid(np.real(F01), grid)
    print("∫F = Γ(0)?  (0,0):", round(I00, 3), "vs", round(G0[0, 0], 3),
          " (1,1):", round(I11, 3), "vs", round(G0[1, 1], 3),
          " (0,1):", round(I01, 3), "vs", round(G0[0, 1], 3))
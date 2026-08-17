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

    # Γ(0) = ∫F (时域方差 = 谱面积, 矩阵逐元)
    grid = np.linspace(-np.pi, np.pi, 4001)
    L00 = np.array([spectral_var1(Phi, Omega, x)[0, 0] for x in grid])
    L11 = np.array([spectral_var1(Phi, Omega, x)[1, 1] for x in grid])
    L01 = np.array([spectral_var1(Phi, Omega, x)[0, 1] for x in grid])
    G0 = np.array([[np.real(np.trapezoid(L00, grid)), np.real(np.trapezoid(L01, grid))],
                   [np.real(np.trapezoid(L01, grid)), np.real(np.trapezoid(L11, grid))]],
                  dtype=float)
    print("Γ(0)=∫F dλ 矩阵:\n", np.round(G0, 3))
    print("Γ(0) 特征值(应>=0):", np.round(np.linalg.eigvalsh(G0), 3))
    # 非对角由 F 的 (0,1) 实部积分给出, 亦对称
    print("Γ(0) 对称: 差 =", np.max(np.abs(G0 - G0.T)))
    print("注: 长程方差 Σλ(h)=2πF(0) 是另一概念(见 10.5), 非 Γ(0)。")
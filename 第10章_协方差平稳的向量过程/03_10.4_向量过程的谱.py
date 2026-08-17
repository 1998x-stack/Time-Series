# 03_10.4 向量过程的谱

"""
Lecture: /第10章 协方差平稳的向量过程
Content: 03_10.4 向量过程的谱
"""

import numpy as np


def spectral_var1(Phi, Omega, lam):
    """VAR(1) 谱矩阵 F(λ) = (1/2π)(I-Φe^{-iλ})^{-1}Ω(I-Φ'e^{iλ})^{-1}。"""
    n = Phi.shape[0]
    ze = np.exp(-1j * lam)
    A = np.linalg.inv(np.eye(n) - Phi * ze)
    B = np.linalg.inv(np.eye(n) - Phi.T / ze)
    return (A @ Omega @ B) / (2 * np.pi)


if __name__ == "__main__":
    Phi = np.array([[0.5, 0.1], [0.2, 0.4]])
    Omega = np.array([[1.0, 0.3], [0.3, 1.0]])

    lam = np.array([0.0, 1.0, np.pi])
    print("F(λ) 对角 (随 λ 变化):")
    for l in lam:
        F = spectral_var1(Phi, Omega, l)
        print(f"  λ={l:.2f}: diag={np.round(np.diag(F),4)}")

    # 积分 = Gamma(0)
    grid = np.linspace(-np.pi, np.pi, 8001)
    diag1 = np.array([spectral_var1(Phi, Omega, x)[0, 0] for x in grid])
    diag2 = np.array([spectral_var1(Phi, Omega, x)[1, 1] for x in grid])
    print("∫F11 =", np.round(np.real(np.trapezoid(diag1, grid)), 3),
          "  ∫F22 =", np.round(np.real(np.trapezoid(diag2, grid)), 3))
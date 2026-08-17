# 05_11.6 向量自回归与结构计量模型

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def irf_var1(Phi, horizon):
    n = Phi.shape[0]
    Psi = np.zeros((horizon + 1, n, n))
    A = np.eye(n)
    for k in range(horizon + 1):
        Psi[k] = A
        A = A @ Phi
    return Psi


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi = np.array([[0.5, 0.2], [0.1, 0.4]])
    Omega = np.array([[1.2, 0.4], [0.4, 0.9]])
    n = 2
    B = np.linalg.cholesky(Omega)          # 结构(下三角)矩阵
    Psi = irf_var1(Phi, 4)

    print("B (Cholesky):\n", np.round(B, 3))
    print("结构冲击样本协方差(应≈I):")
    L = B
    burn = 200; total = 30000
    e = np.zeros((total, 2))
    Y = np.zeros((total, 2))
    for t in range(1, total):
        Y[t] = Phi @ Y[t - 1] + L @ rng.normal(size=n)
        # ε = B e  => e = B^{-1} ε (略)
    # e_t = B^{-1} ε_t, 用残差近似
    resid = Y[1:] - Y[:-1] @ Phi.T
    estruc = np.linalg.solve(B, resid.T).T     # 结构冲击
    print(np.round(np.cov(estruc.T), 3))
    # 结构 IRF
    sIRF = np.stack([Psi[k] @ B for k in range(3)])
    print("结构 IRF Ψ_k B (k=0..2):\n", np.round(sIRF, 3))
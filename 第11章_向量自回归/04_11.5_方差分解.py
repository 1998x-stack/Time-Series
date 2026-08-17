# 04_11.5 方差分解

import numpy as np


def fevd_var1(Phi, Omega, horizon):
    """预测误差方差分解份额: (horizon, n, n)——第h期, i行对j冲击份额。"""
    n = Phi.shape[0]
    P = np.linalg.cholesky(Omega)
    shares = np.zeros((horizon + 1, n, n))
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        Psi_k = np.eye(n)
        for k in range(h):
            acc += (Psi_k @ P) ** 2
            Psi_k = Psi_k @ Phi
        denom = acc.sum(axis=1)
        shares[h] = acc / denom[:, None]
    return shares


if __name__ == "__main__":
    Phi = np.array([[0.5, 0.3], [0.0, 0.4]])
    Omega = np.array([[1.0, 0.2], [0.2, 1.0]])
    shares = fevd_var1(Phi, Omega, 6)
    print("h=1 拆分 (行=变量, 列=冲击):\n", np.round(shares[1], 3))
    print("h=6 拆分:\n", np.round(shares[6], 3))
    print("行和(应为1):", np.round(shares[6].sum(1), 3))
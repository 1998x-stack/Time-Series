# 03_11.4 脉冲响应函数

import numpy as np


def irf_var1(Phi, horizon):
    """VAR(1) 脉冲响应矩阵 Psi_k = Phi^k, k=0..horizon。"""
    n = Phi.shape[0]
    Psi = np.zeros((horizon + 1, n, n))
    A = np.eye(n)
    for k in range(horizon + 1):
        Psi[k] = A
        A = A @ Phi
    return Psi


if __name__ == "__main__":
    Phi = np.array([[0.5, 0.1], [0.2, 0.3]])
    Psi = irf_var1(Phi, 5)
    print("脉冲响应 (i,j = 冲击j→变量i):")
    for k in range(4):
        print(f"  Ψ{k}: {np.round(Psi[k], 3)}")

    # 对照: 确定性模拟单位冲击 e1
    n = 2; H = 5
    e0 = np.array([1.0, 0.0])
    y = np.zeros((H + 1, n))
    z = e0.copy(); y[0] = z
    for t in range(1, H + 1):
        z = Phi @ z
        y[t] = z
    print("模拟单位冲击路径 y1:", np.round(y[:, 0], 4))
    print("理论 Ψ_k 第1列:", np.round(Psi[: H + 1, :, 0][:, 0], 4))
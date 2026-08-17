# 01_1.2 p阶差分方程

"""
Lecture: /第1章 差分方程
Content: 01_1.2 p阶差分方程
"""

import numpy as np


def companion_matrix(phi: np.ndarray) -> np.ndarray:
    """返回 p 阶差分方程的伴随矩阵 F (第1行=系数, 其余为单位移块)。

    Args:
        phi: 长度 p 的系数向量。

    Returns:
        shape (p, p) 的相伴矩阵。
    """
    p = len(phi)
    F = np.zeros((p, p))
    F[0, :] = phi
    if p > 1:
        F[1:, : p - 1] = np.eye(p - 1)
    return F


def forward_solve(phi: np.ndarray, y_init: np.ndarray, w: np.ndarray) -> np.ndarray:
    """相伴矩阵法状态递推 z_t = F z_{t-1} + (w_t, 0, ..., 0)'。

    Args:
        phi: 系数向量。
        y_init: 初始状态 z 的初值(长 p)。
        w: 扰动序列。

    Returns:
        shape (T,) 的 y 序列 (各时刻状态首分量)。
    """
    p = len(phi)
    F = companion_matrix(phi)
    T = len(w)
    y = np.empty(T)
    z = y_init.copy()
    for t in range(T):
        z = F @ z
        z[0] += w[t]
        y[t] = z[0]
    return y


def direct_solve(phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """直接用 p 系数递推 y_t = phi1*y_{t-1} + ... + phip*y_{t-p} + w_t (零初值)。

    Args:
        phi: 系数向量。
        w: 扰动序列。

    Returns:
        shape (T,) 的解序列。
    """
    p = len(phi)
    T = len(w)
    y = np.empty(T)
    for t in range(T):
        y[t] = w[t] + sum(phi[j] * (y[t - 1 - j] if t - 1 - j >= 0 else 0.0) for j in range(p))
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi = np.array([0.4, -0.2])
    p = len(phi)
    T = 300

    F = companion_matrix(phi)
    ev = np.linalg.eigvals(F)
    print("特征值:", np.round(ev, 4), "模:", np.round(np.abs(ev), 4),
          "稳定:", bool(np.max(np.abs(ev)) < 1))

    w = rng.normal(size=T)
    # 两法均以零初始条件出发, 以便精确比对两法等价性
    a = forward_solve(phi, np.zeros(p), w)
    b = direct_solve(phi, w)
    print(f"相伴矩阵 vs 直接递推 max-diff = {np.max(np.abs(a - b)):.3e}")
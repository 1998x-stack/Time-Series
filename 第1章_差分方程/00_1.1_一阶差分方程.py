# 00_1.1 一阶差分方程

"""
Lecture: /第1章 差分方程
Content: 00_1.1 一阶差分方程
"""

import numpy as np


def recurse_solve(phi: float, x0: float, w: np.ndarray) -> np.ndarray:
    """向前递推解一阶差分方程 y_t = phi*y_{t-1} + w_t。

    Args:
        phi: 系数。
        x0: 初值。
        w: 扰动序列 w[0..T-1]。

    Returns:
        shape (T,) 的解序列 y[0..T-1]。
    """
    x = np.empty_like(w)
    prev = x0
    for t in range(len(w)):
        prev = phi * prev + w[t]
        x[t] = prev
    return x


def closed_form(phi: float, x0: float, w: np.ndarray) -> np.ndarray:
    """显式解 y_t = phi^{t+1}x0 + sum_{j=0..t} phi^{t-j} w_j (0 起)。

    Args:
        phi: 系数。
        x0: 初值。
        w: 扰动序列 w[0..T-1]。

    Returns:
        shape (T,) 的显式解序列。
    """
    T = len(w)
    impact = phi ** np.arange(1, T + 1) * x0
    psi = phi ** np.arange(T)          # Green 函数(脉冲响应)
    past = np.convolve(w, psi)[:T]     # sum_{j<=t} psi_{t-j} w_j
    return impact + past


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    x0 = 1.0
    for phi in (0.6, 0.9, 1.0, 1.05):
        w = rng.normal(size=T)
        a = recurse_solve(phi, x0, w)
        b = closed_form(phi, x0, w)
        print(f"[phi={phi}] 递推 vs 闭式 max-diff = {np.max(np.abs(a-b)):.3e} | "
              f"初值分量 phi^T = {phi**T:.6e}")
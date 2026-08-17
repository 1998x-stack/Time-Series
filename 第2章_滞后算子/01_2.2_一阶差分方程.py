# 01_2.2 一阶差分方程

"""
Lecture: /第2章 滞后算子
Content: 01_2.2 一阶差分方程
"""

import numpy as np


def lag(x: np.ndarray, k: int = 1) -> np.ndarray:
    """滞后算子: 返回左移 k 期、右侧补零的序列。"""
    n = len(x)
    if k >= n:
        return np.zeros(n)
    out = np.zeros(n)
    out[k:] = x[: n - k]
    return out


def recursive_solve(phi: float, w: np.ndarray) -> np.ndarray:
    """递推 y_t = phi*y_{t-1} + w_t (从 y_0=w_0)。"""
    y = np.empty_like(w)
    prev = 0.0
    for t in range(len(w)):
        prev = phi * prev + w[t]
        y[t] = prev
    return y


def inverse_solve(phi: float, J: int, w: np.ndarray) -> np.ndarray:
    """截断逆算子 sum_{j=0..J} phi^j L^j 作用于 w → y。"""
    total = np.zeros_like(w)
    for j in range(J + 1):
        total += phi ** j * lag(w, j)
    return total


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 120
    phi = 0.6
    w = rng.normal(size=n)

    y = recursive_solve(phi, w)

    # (1 - phi L) y 应精确还原 w
    resid = y - phi * lag(y)
    print("递推解: (1 - phiL)y 与 w 最大差 =", np.max(np.abs(resid - w)))

    # 截断逆逼近递推解, 误差随 J 递减 (≈ phi^(J+1))
    print("截断逆 vs 递推解 最大差:")
    for J in (2, 4, 8, 16):
        y_inv = inverse_solve(phi, J, w)
        err = np.max(np.abs(y_inv - y))
        print(f"  J={J:2d}  差 = {err:.3e}  量级 phi^(J+1)={phi**(J+1):.2e}")
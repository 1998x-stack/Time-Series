# 00_2.1 简介

"""
Lecture: /第2章 滞后算子
Content: 00_2.1 简介
"""

import numpy as np


def lag(x: np.ndarray, k: int = 1) -> np.ndarray:
    """滞后算子 L^k 作用于数组: 返回左移 k 期、右侧补零的序列。"""
    n = len(x)
    if k >= n:
        return np.zeros(n)
    out = np.zeros(n)
    out[k:] = x[: n - k]
    return out


def truncated_inverse(phi: float, J: int, w: np.ndarray) -> np.ndarray:
    """截断逆算子 sum_{j=0..J} phi^j L^j 作用于 w。"""
    total = np.zeros_like(w)
    for j in range(J + 1):
        total += phi ** j * lag(w, j)
    return total


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 60
    phi = 0.5
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    a, b = 2.0, -0.5

    # 线性与时移不变性: L(a x + b y) == a L(x) + b L(y)
    linear_lhs = lag(a * x + b * y)
    linear_rhs = a * lag(x) + b * lag(y)
    shift = lag(lag(x))          # L^2 x = L(Lx)
    print("线性性  max-diff:", np.max(np.abs(linear_lhs - linear_rhs)))
    print("L^2=L∘L max-diff:", np.max(np.abs(shift - lag(x, 2))))

    # 几何级数逆: (1 - phi L) * sum_{j=0..J} phi^j L^j (w) ≈ w
    w = rng.normal(size=n)
    print("截断逆误差 (随 J 递减, 约 phi^(J+1)):")
    for J in (2, 4, 8, 16):
        z = truncated_inverse(phi, J, w)
        resid = z - phi * lag(z)          # (1 - phi L) z
        err = np.max(np.abs(resid[: n - 1] - w[: n - 1]))   # 忽略最末边界
        print(f"  J={J:2d}  误差 = {err:.3e}   理论量级 phi^(J+1)={phi**(J+1):.2e}")
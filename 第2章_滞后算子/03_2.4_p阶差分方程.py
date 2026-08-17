# 03_2.4 p阶差分方程

"""
Lecture: /第2章 滞后算子
Content: 03_2.4 p阶差分方程
"""

import numpy as np
from itertools import combinations


def lag(x: np.ndarray, k: int = 1) -> np.ndarray:
    """滞后算子: 返回左移 k 期、右侧补零的序列(保留 dtype)。"""
    n = len(x)
    if k >= n:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[k:] = x[: n - k]
    return out


def apply_full(phi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """一步算子 (1 - phi1 L - ... - phip L^p) x。"""
    total = x.copy()
    for i, p in enumerate(phi, start=1):
        total -= p * lag(x, i)
    return total


def apply_factors(lams: np.ndarray, x: np.ndarray) -> np.ndarray:
    """依序作用 p 个一次因子 prod (1 - lam_j L) x。"""
    for lam in lams:
        x = x - lam * lag(x)
    return x


def elementary_sym(lams: np.ndarray, k: int) -> complex:
    """第 k 个初等对称和 e_k = sum_{S:|S|=k} prod_{j in S} lam_j。"""
    return sum(np.prod(np.take(lams, idx)) for idx in combinations(range(len(lams)), k))


def recursive_solve(phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """递推 y_t = sum_i phi_i y_{t-i} + w_t (零初值)。"""
    p = len(phi)
    n = len(w)
    y = np.empty(n)
    for t in range(n):
        val = w[t]
        for i in range(p):
            if t - 1 - i >= 0:
                val += phi[i] * y[t - 1 - i]
        y[t] = val
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    p = 4
    # 稳定的 AR(4) 系数 (特征值模<1)
    phi = np.array([0.4, -0.3, 0.15, -0.03])
    F = np.zeros((p, p)); F[0, :] = phi
    if p > 1:
        F[1:, : p - 1] = np.eye(p - 1)
    lam = np.linalg.eigvals(F)
    print("特征值模:", np.round(np.abs(lam), 4), " 稳定:", np.all(np.abs(lam) < 1))

    # 系数与初等对称和关系 phi_k = (-1)^{k+1} e_k
    phi_rec = np.array([(-1) ** (kk + 1) * elementary_sym(lam, kk)
                        for kk in range(1, p + 1)])
    print("由特征根恢复系数:", np.round(np.real(phi_rec), 4))
    print("真值系数:        ", phi)

    # 逐步因子作用 == 一步 p 阶算子
    w = rng.normal(size=80)
    one = apply_full(phi, w)
    two = apply_factors(lam, w)
    print("逐步因子 vs 一步算子 max-diff:", np.max(np.abs(two - one)))

    # 递推解代回原方程
    y = recursive_solve(phi, w)
    resid = apply_full(phi, y)
    print("(1-sum phi_i L^i)y vs w max-diff:", np.max(np.abs(resid - w)))
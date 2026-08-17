# 02_2.3 二阶差分方程

"""
Lecture: /第2章 滞后算子
Content: 02_2.3 二阶差分方程
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


def apply_full(phi1, phi2, x):
    """一步算子 (1 - phi1 L - phi2 L^2) x。"""
    return x - phi1 * lag(x) - phi2 * lag(x, 2)


def apply_factor(lamb, x):
    """一次因子 (1 - lamb L) x。"""
    return x - lamb * lag(x)


def recursive_solve(phi1, phi2, w):
    """递推 y_t = phi1 y_{t-1} + phi2 y_{t-2} + w_t (零初值)。"""
    n = len(w)
    y = np.empty(n)
    for t in range(n):
        prev = 0.0
        if t - 1 >= 0:
            prev += phi1 * y[t - 1]
        if t - 2 >= 0:
            prev += phi2 * y[t - 2]
        y[t] = prev + w[t]
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi1, phi2 = 0.5, -0.06
    n = 100

    # 特征值: 相伴矩阵
    F = np.array([[phi1, phi2], [1.0, 0.0]])
    lam = np.linalg.eigvals(F)
    print("特征值 lam:", np.round(lam, 4), " 模<1:", np.all(np.abs(lam) < 1))
    print("lam1+lam2 =", round(lam.sum(), 4), "≈ phi1 =", phi1)
    print("lam1*lam2 =", round(lam.prod(), 4), "≈ -phi2 =", -phi2)

    # 因式分解展开系数匹配
    c1 = lam.sum()      # phi1
    c2 = -lam.prod()    # phi2
    print("因式分解恢复系数:", (round(c1, 6), round(c2, 6)),
          "与真值:", (phi1, phi2))

    # 两步算子 == 一步算子
    w = rng.normal(size=n)
    two_step = apply_factor(lam[1], apply_factor(lam[0], w))
    one_step = apply_full(phi1, phi2, w)
    print("两步 vs 一步算子 max-diff:", np.max(np.abs(two_step - one_step)))

    # 递推解:'y 由 (1 - phi1 L - phi2 L^2) 逆得来, 代回原方程应得 w
    y = recursive_solve(phi1, phi2, w)
    resid = apply_full(phi1, phi2, y)
    print("(1-phi1L-phi2L^2)y vs w max-diff:", np.max(np.abs(resid - w)))
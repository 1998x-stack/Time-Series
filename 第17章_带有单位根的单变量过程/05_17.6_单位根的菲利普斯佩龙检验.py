# 05_17.6 单位根的菲利普斯佩龙(PP)检验

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def newey_west_sq(u, m):
    """Newey-West 长程方差: gamma0 + 2 sum (1-j/(m+1)) gamma_j。"""
    T = len(u); uc = u - u.mean()
    g = np.array([np.dot(uc[h:], uc[: T - h]) / T for h in range(m + 1)])
    return g[0] + 2 * sum((1 - j / (m + 1)) * g[j] for j in range(1, m + 1))


def pp_rho(y, m=8):
    """PP ρ 统计(Phillips-Perron): Z_rho = T(ρ̂-1) - (T²(λ̂²-σ̂²))/(2 Σ y_{t-1}²)。"""
    T = len(y)
    X = y[:-1]
    rho = np.dot(X, y[1:]) / np.dot(X, X)
    u = y[1:] - rho * X
    sig2 = np.sum(u ** 2) / (T - 1)
    lam2 = newey_west_sq(u, m)
    denom = 2 * (X ** 2).sum()
    Z_rho = T * (rho - 1) - (T ** 2 * (lam2 - sig2)) / denom
    return Z_rho, sig2, lam2


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 1000
    # 带一阶自相关误差的单位根: y_t = y_{t-1} + u_t, u_t=0.5u_{t-1}+e_t
    e = rng.normal(0, 1, T)
    u = np.zeros(T); y = np.zeros(T)
    for t in range(1, T):
        u[t] = 0.5 * u[t - 1] + e[t]
        y[t] = y[t - 1] + u[t]

    Z_rho, sig2, lam2 = pp_rho(y)
    print(f"σ̂² = {sig2:.3f}   长程 λ̂² = {lam2:.3f}   (PP 用 λ̂², 大于 σ̂²)")
    print(f"PP ρ 统计 Z_rho = {Z_rho:.2f}")
    print("临界值取 DF 表(非标准); 以 Z_rho 与 DF 界比较判定单位根")
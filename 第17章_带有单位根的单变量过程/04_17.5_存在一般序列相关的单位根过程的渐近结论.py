# 04_17.5 存在一般序列相关的单位根过程的渐近结论

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200000
    # u_t = 0.6 u_{t-1} + eta (自相关误差)
    e = rng.normal(0, 1, T)
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = 0.6 * u[t - 1] + e[t]

    sig2 = np.var(u)
    # 长程方差: sum_h gamma(h) = gamma(0)(1+rho)/(1-rho)
    lam2 = sig2 * (1 + 0.6) / (1 - 0.6)
    print(f"σ²(样本) = {sig2:.3f}   长程 λ² = σ²(1+ρ)/(1-ρ) = {lam2:.3f}")
    print("自相关加大长程方差; 单位根检验时须用 λ² 修正(PP/DF-HAC)")
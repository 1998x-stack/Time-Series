# 08_17.9 贝叶斯分析

import numpy as np
from scipy import stats


def posterior_rho(y):
    """条件高斯下 rho 的后验均值与方差(平坦先验)。"""
    X = y[:-1]; yy = y[1:]
    var = None
    a = np.dot(X, X); b = np.dot(X, yy)
    mean = b / a
    # 用 OLS 残差估 sigma^2
    e = yy - mean * X
    sig2 = np.sum(e ** 2) / (len(yy) - 1)
    varr = sig2 / a
    return mean, varr


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 2000
    for name, rho in (("近稳", 0.9), ("单位根", 1.0)):
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = rho * y[t - 1] + rng.normal(0, 1)
        m, v = posterior_rho(y)
        p_space = 1 - stats.norm.cdf(1.0, m, np.sqrt(v))   # P(rho>=1)
        print(f"{name}: 后验均值 ρ={m:.4f}, SD={np.sqrt(v):.4f}, "
              f"P(ρ≥1)={p_space:.3f}")
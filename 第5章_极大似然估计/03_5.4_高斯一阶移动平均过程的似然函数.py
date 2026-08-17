# 03_5.4 高斯一阶移动平均过程的似然函数

"""
Lecture: /第5章 极大似然估计
Content: 03_5.4 高斯一阶移动平均过程的似然函数
"""

import numpy as np
from scipy.optimize import minimize


def simulate_ma1(theta, sigma, n, rng):
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    pe = 0.0
    for t in range(total):
        x[t] = eps[t] + theta * pe
        pe = eps[t]
    return x[burn:]


def neg_loglik_ma1(par, x):
    """MA(1) 高斯对数似然(创新递归, 负号)。"""
    theta, sig2 = par
    if sig2 <= 0:
        return 1e18
    n = len(x)
    ll = n / 2 * np.log(2 * np.pi * sig2)
    pe = 0.0
    for xt in x:
        e = xt - theta * pe
        ll += e ** 2 / (2 * sig2)
        pe = e
    return ll


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    theta0, sig0 = 0.5, 1.0
    n = 4000
    x = simulate_ma1(theta0, sig0, n, rng)

    res = minimize(lambda p: neg_loglik_ma1(p, x),
                   x0=[0.0, 1.0], method="Nelder-Mead")
    print("MLE: theta =", round(res.x[0], 4), "  sigma^2 =", round(res.x[1], 4))
    print("真值: theta =", theta0, "  sigma^2 =", sig0)
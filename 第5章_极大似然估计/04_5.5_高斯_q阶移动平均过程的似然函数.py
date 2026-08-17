# 04_5.5 高斯_q阶移动平均过程的似然函数

"""
Lecture: /第5章 极大似然估计
Content: 04_5.5 高斯_q阶移动平均过程的似然函数
"""

import numpy as np
from scipy.optimize import minimize


def simulate_ma(theta, sigma, n, rng):
    q = len(theta)
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(total):
        val = eps[t]
        for j in range(q):
            if t - 1 - j >= 0:
                val += theta[j] * eps[t - 1 - j]
        x[t] = val
    return x[burn:]


def neg_loglik_maq(par, x, q):
    theta, sig2 = par[:q], par[q]
    if sig2 <= 0:
        return 1e30
    n = len(x)
    ll = n / 2 * np.log(2 * np.pi * sig2)
    e = np.zeros(max(q, n))
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(n):
            e[t] = x[t] - sum(theta[j] * e[t - 1 - j] for j in range(q) if t - 1 - j >= 0)
            ll += e[t] ** 2 / (2 * sig2)
    if not np.isfinite(ll):
        return 1e30
    return ll


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    theta0 = np.array([0.5, -0.3])
    q = len(theta0)
    sigma = 1.0
    n = 5000
    x = simulate_ma(theta0, sigma, n, rng)

    res = minimize(lambda p: neg_loglik_maq(p, x, q),
                   x0=[0.0, 0.0, 1.0], method="Nelder-Mead")
    print("MLE: theta =", np.round(res.x[:q], 4), "  sigma^2 =", round(res.x[q], 4))
    print("真值: theta =", theta0, "  sigma^2 =", sigma)
# 05_5.6 高斯_p阶自回归q阶移动平均过程的似然函数

"""
Lecture: /第5章 极大似然估计
Content: 05_5.6 高斯_p阶自回归q阶移动平均过程的似然函数
"""

import numpy as np
from scipy.optimize import minimize


def simulate_arma(phi, theta, sigma, n, rng):
    p, q = len(phi), len(theta)
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(total):
        val = eps[t]
        for i in range(p):
            if t - 1 - i >= 0:
                val += phi[i] * x[t - 1 - i]
        for j in range(q):
            if t - 1 - j >= 0:
                val += theta[j] * eps[t - 1 - j]
        x[t] = val
    return x[burn:]


def neg_loglik_arma(par, x, p, q):
    phi, theta, sig2 = par[:p], par[p:p + q], par[p + q]
    if sig2 <= 0:
        return 1e30
    n = len(x)
    ll = n / 2 * np.log(2 * np.pi * sig2)
    xpad = np.zeros(n + p)
    xpad[:n] = x
    e = np.zeros(n + q)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(n):
            e[t] = x[t]
            for i in range(p):
                if t - 1 - i >= 0:
                    e[t] -= phi[i] * x[t - 1 - i]
            for j in range(q):
                if t - 1 - j >= 0:
                    e[t] -= theta[j] * e[t - 1 - j]
            ll += e[t] ** 2 / (2 * sig2)
    if not np.isfinite(ll):
        return 1e30
    return ll


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi0 = np.array([0.6])
    theta0 = np.array([0.4])
    p, q = 1, 1
    sigma = 1.0
    n = 5000
    x = simulate_arma(phi0, theta0, sigma, n, rng)

    res = minimize(lambda prm: neg_loglik_arma(prm, x, p, q),
                   x0=[0.0, 0.0, 1.0], method="Nelder-Mead")
    print("MLE: phi =", round(res.x[0], 4), " theta =", round(res.x[1], 4),
          " sigma^2 =", round(res.x[2], 4))
    print("真值: phi =", phi0[0], " theta =", theta0[0], " sigma^2 =", sigma)
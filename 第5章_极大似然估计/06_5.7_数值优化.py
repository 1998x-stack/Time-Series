# 06_5.7 数值优化

"""
Lecture: /第5章 极大似然估计
Content: 06_5.7 数值优化
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
    e = np.zeros(n)
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
    return 1e30 if not np.isfinite(ll) else ll


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi0, theta0 = np.array([0.6]), np.array([0.4])
    x = simulate_arma(phi0, theta0, 1.0, 4000, rng)
    x0 = [0.0, 0.0, 1.0]

    for method in ("Nelder-Mead", "BFGS"):
        res = minimize(lambda p: neg_loglik_arma(p, x, 1, 1), x0,
                       method=method)
        print(f"{method:12s}: phi={res.x[0]:.4f} theta={res.x[1]:.4f} "
              f"sig2={res.x[2]:.4f}  func={res.fun:.2f}  nfev={res.nfev}  "
              f"成功={res.success}")
    print("真值: phi=0.6 theta=0.4 sig2=1.0")
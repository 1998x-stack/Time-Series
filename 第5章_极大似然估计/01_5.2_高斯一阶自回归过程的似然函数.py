# 01_5.2 高斯一阶自回归过程的似然函数

"""
Lecture: /第5章 极大似然估计
Content: 01_5.2 高斯一阶自回归过程的似然函数
"""

import numpy as np
from scipy.optimize import minimize


def simulate_ar1(phi, sigma, n, rng):
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(1, total):
        x[t] = phi * x[t - 1] + eps[t]
    return x[burn:]


def neg_loglik(params, x):
    phi, sig2 = params
    if sig2 <= 0:
        return 1e18
    n = len(x)
    e = x[1:] - phi * x[:-1]
    return (n - 1) / 2 * np.log(2 * np.pi * sig2) + np.sum(e ** 2) / (2 * sig2)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi0, sig0 = 0.6, 1.0
    n = 3000
    x = simulate_ar1(phi0, sig0, n, rng)

    # 闭式
    num = np.dot(x[1:], x[:-1]); den = np.dot(x[:-1], x[:-1])
    phi_hat = num / den
    e = x[1:] - phi_hat * x[:-1]
    sig_hat = np.mean(e ** 2)
    print("闭式 MLE: phi =", round(phi_hat, 4), " sigma^2 =", round(sig_hat, 4))

    # 数值 MLE
    res = minimize(lambda p: neg_loglik(p, x), x0=[0.0, 1.0], method="Nelder-Mead")
    print("数值 MLE: phi =", round(res.x[0], 4), " sigma^2 =", round(res.x[1], 4))
    print("真值:      phi =", phi0, "  sigma^2 =", sig0)
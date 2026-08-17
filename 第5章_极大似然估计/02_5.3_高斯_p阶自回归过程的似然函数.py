# 02_5.3 高斯_p阶自回归过程的似然函数

"""
Lecture: /第5章 极大似然估计
Content: 02_5.3 高斯_p阶自回归过程的似然函数
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")
from scipy.optimize import minimize


def simulate_ar(phi, sigma, n, rng):
    p = len(phi)
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(total):
        val = eps[t]
        for i in range(p):
            if t - 1 - i >= 0:
                val += phi[i] * x[t - 1 - i]
        x[t] = val
    return x[burn:]


def neg_loglik_ar(params, x, p):
    phi, sig = params[:p], params[p]
    if sig <= 0:
        return 1e18
    n = len(x)
    e = np.empty(n - p)
    for t in range(p, n):
        e[t - p] = x[t] - sum(phi[i] * x[t - 1 - i] for i in range(p))
    return (n - p) / 2 * np.log(2 * np.pi * sig) + np.sum(e ** 2) / (2 * sig)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi0 = np.array([0.5, -0.2])
    p = len(phi0)
    sigma = 1.0
    n = 3000
    x = simulate_ar(phi0, sigma, n, rng)

    # OLS: 回归 y_t = x_t 对滞后列 [x_{t-1}, x_{t-2}]
    X = np.column_stack([x[p - 1 - i: n - 1 - i] for i in range(p)])
    # 校验: 每列长度为 n-p
    y = x[p:]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sig_ols = np.mean(resid ** 2)
    print("OLS MLE: phi =", np.round(beta, 4), "  sigma^2 =", round(sig_ols, 4))

    # 数值 MLE
    res = minimize(lambda q: neg_loglik_ar(q, x, p),
                   x0=[0.0, 0.0, 1.0], method="Nelder-Mead")
    print("数值 MLE: phi =", np.round(res.x[:p], 4), "  sigma^2 =", round(res.x[p], 4))
    print("真值:     phi =", phi0, "  sigma^2 =", sigma)
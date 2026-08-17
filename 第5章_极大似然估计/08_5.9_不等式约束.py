# 08_5.9 不等式约束

"""
Lecture: /第5章 极大似然估计
Content: 08_5.9 不等式约束
"""

import numpy as np
from scipy.optimize import minimize


def neg_loglik_ma1(par, x):
    theta, sig2 = par
    if sig2 <= 0:
        return 1e30
    n = len(x)
    ll = n / 2 * np.log(2 * np.pi * sig2)
    pe = 0.0
    for xt in x:
        e = xt - theta * pe
        ll += e ** 2 / (2 * sig2)
        pe = e
    return ll if np.isfinite(ll) else 1e30


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 4000
    eps = rng.normal(0.0, 1.0, n)
    x = np.zeros(n)
    pe = 0.0
    for t in range(n):
        x[t] = eps[t] + 0.5 * pe
        pe = eps[t]

    # 无约束
    r1 = minimize(lambda p: neg_loglik_ma1(p, x), [0.0, 1.0], method="Nelder-Mead")
    # 有约束: theta in (-1,1), sig2 in (1e-6, inf)
    r2 = minimize(lambda p: neg_loglik_ma1(p, x), [0.0, 1.0],
                  method="L-BFGS-B",
                  bounds=[(-0.99, 0.99), (1e-6, np.inf)])
    print("无约束 MLE: theta =", round(r1.x[0], 4), " sig2 =", round(r1.x[1], 4))
    print("有约束 MLE: theta =", round(r2.x[0], 4), " sig2 =", round(r2.x[1], 4),
          " (|theta|<1 保持)")
    print("真值: theta = 0.5, sig2 = 1.0")
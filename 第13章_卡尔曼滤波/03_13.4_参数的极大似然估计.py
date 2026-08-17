# 03_13.4 参数的极大似然估计

import numpy as np
from scipy.optimize import minimize


def neg_ll(par, y):
    """状态空间(AR1+noise)预测误差负对数似然。par=(phi,q,r)。"""
    phi, q, r = par
    if q <= 0 or r <= 0:
        return 1e18
    xi = 0.0; P = 10.0
    ll = 0.0
    for t in range(len(y)):
        xip = phi * xi; Pp = phi ** 2 * P + q
        S = Pp + r
        eta = y[t] - xip
        ll += 0.5 * (np.log(S) + eta ** 2 / S)
        K = Pp / S
        xi = xip + K * eta; P = Pp - K * Pp
    return ll


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi0, q0, r0 = 0.8, 1.0, 0.5
    T = 4000
    xi = np.zeros(T); y = np.zeros(T)
    for t in range(T):
        if t > 0:
            xi[t] = phi0 * xi[t - 1] + rng.normal(0, np.sqrt(q0))
        y[t] = xi[t] + rng.normal(0, np.sqrt(r0))

    res = minimize(lambda p: neg_ll(p, y), np.array([0.0, 1.0, 1.0]),
                   method="Nelder-Mead", options=dict(maxiter=10000))
    est = res.x
    print(f"MLE: phi={est[0]:.3f} q={est[1]:.3f} r={est[2]:.3f}")
    print(f"真值: phi={phi0} q={q0} r={r0}")
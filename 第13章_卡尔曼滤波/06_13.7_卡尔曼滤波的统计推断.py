# 06_13.7 卡尔曼滤波的统计推断

import numpy as np


def run_filter_with_innov(y, phi, q, r):
    T = len(y)
    xi = 0.0; P = 10.0
    eta = np.zeros(T); S = np.zeros(T)
    for t in range(T):
        xip = phi * xi; Pp = phi ** 2 * P + q
        S[t] = Pp + r
        eta[t] = y[t] - xip
        K = Pp / S[t]
        xi = xip + K * eta[t]; P = Pp - K * Pp
    return eta, S


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi, q, r = 0.8, 1.0, 0.5
    T = 10000
    xi = np.zeros(T); y = np.zeros(T)
    for t in range(T):
        if t > 0:
            xi[t] = phi * xi[t - 1] + rng.normal(0, np.sqrt(q))
        y[t] = xi[t] + rng.normal(0, np.sqrt(r))

    eta, S = run_filter_with_innov(y, phi, q, r)
    nu = eta / np.sqrt(S)                    # 标准化
    print("标准化新息 方差:", round(nu.var(), 3), " (应≈1)")
    acf = np.array([np.corrcoef(nu[h:], nu[:-h])[0, 1] for h in range(1, 4)])
    print("新息 ACF (h=1..3):", np.round(acf, 4), " (应≈0, 白)")
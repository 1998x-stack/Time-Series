# 05_13.6 平滑

import numpy as np


def kalman_run(y, phi, q, r):
    """正向滤波, 返回 xif, xip, Pp, P11。"""
    T = len(y)
    xi = 0.0; P = 10.0
    xif = np.zeros(T); xip_a = np.zeros(T); Pp_a = np.zeros(T); P11_a = np.zeros(T)
    for t in range(T):
        xip = phi * xi; Pp = phi ** 2 * P + q
        S = Pp + r
        K = Pp / S
        xi = xip + K * (y[t] - xip)
        P = Pp - K * Pp
        xif[t] = xi; xip_a[t] = xip; Pp_a[t] = Pp; P11_a[t] = P
    return xif, xip_a, Pp_a, P11_a


def rts_smooth(xif, xip_a, Pp_a, P11_a, phi):
    T = len(xif)
    xs = np.zeros(T); xs[T - 1] = xif[T - 1]
    for t in range(T - 2, -1, -1):
        J = phi * P11_a[t] / Pp_a[t + 1]
        xs[t] = xif[t] + J * (xs[t + 1] - xip_a[t + 1])
    return xs


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi, q, r = 0.8, 1.0, 0.5
    T = 400
    xi_true = np.zeros(T); y = np.zeros(T)
    for t in range(T):
        if t > 0:
            xi_true[t] = phi * xi_true[t - 1] + rng.normal(0, np.sqrt(q))
        y[t] = xi_true[t] + rng.normal(0, np.sqrt(r))

    xif, xip_a, Pp_a, P11_a = kalman_run(y, phi, q, r)
    xs = rts_smooth(xif, xip_a, Pp_a, P11_a, phi)
    print("滤波 MSE:", round(np.mean((xif - xi_true) ** 2), 4))
    print("平滑 MSE:", round(np.mean((xs - xi_true) ** 2), 4), " (应≤滤波)")
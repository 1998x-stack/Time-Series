# 08_附录13.A 第13章性质证明

import numpy as np


def kalman(y, phi, q, r):
    T = len(y)
    xi = 0.0; P = 10.0
    xif = np.zeros(T); Pf = np.zeros(T)
    for t in range(T):
        xip = phi * xi; Pp = phi ** 2 * P + q
        S = Pp + r; K = Pp / S
        xi = xip + K * (y[t] - xip)
        P = Pp - K * Pp
        xif[t] = xi; Pf[t] = P
    return xif, Pf


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi, q, r = 0.8, 1.0, 0.5
    T = 200
    B = 4000
    err = np.zeros(T); Pavg = np.zeros(T)
    for b in range(B):
        xi = np.zeros(T); y = np.zeros(T)
        for t in range(T):
            if t > 0:
                xi[t] = phi * xi[t - 1] + rng.normal(0, np.sqrt(q))
            y[t] = xi[t] + rng.normal(0, np.sqrt(r))
        xif, P = kalman(y, phi, q, r)
        err += (xif - xi) ** 2
        Pavg += P
    err /= B; Pavg /= B
    for t in (30, 80, 150):
        print(f"t={t}: 经验 E[(ξ̂-ξ)²]={err[t]:.3f}   平均 P_tt={Pavg[t]:.3f}")
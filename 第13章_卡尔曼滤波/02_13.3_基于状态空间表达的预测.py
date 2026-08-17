# 02_13.3 基于状态空间表达的预测

import numpy as np


def kalman_filter(y, F, Q, H, R, xi0, P0):
    T = len(y); n = F.shape[0]
    xif = np.zeros((T, n)); Pf = np.zeros((T, n, n))
    xi = xi0.copy(); P = P0.copy()
    for t in range(T):
        xip = F @ xi; Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        eta = np.array([[y[t]]]) - H @ xip
        K = Pp @ H.T @ np.linalg.inv(S)
        xi = xip + K @ eta
        P = Pp - K @ H @ Pp
        xif[t] = xi.ravel(); Pf[t] = P
    return xif, Pf


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi, q, r = 0.8, 1.0, 0.5
    F = np.array([[phi]]); Q = np.array([[q]]); H = np.array([[1.0]]); R = np.array([[r]])
    T = 300
    xi = np.zeros(T); y = np.zeros(T)
    for t in range(T):
        if t > 0:
            xi[t] = phi * xi[t - 1] + rng.normal(0, np.sqrt(q))
        y[t] = xi[t] + rng.normal(0, np.sqrt(r))

    xif, P = kalman_filter(y, F, Q, H, R, np.zeros((1, 1)), np.eye(1) * 10)

    xiT = xif[-1]
    Hh = 5
    yf = np.array([(H @ np.linalg.matrix_power(F, m) @ xiT).item()
                   for m in range(1, Hh + 1)])
    print("末端滤波状态 ξT =", round(xif[-1, 0], 3))
    print("未来真实值:", np.round(xi[-Hh:], 3))
    print("多步预测  :", np.round(yf, 3))
    print("(预测误差方差随 h 累积状态噪声)")
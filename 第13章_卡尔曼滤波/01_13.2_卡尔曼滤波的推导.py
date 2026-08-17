# 01_13.2 卡尔曼滤波的推导

import numpy as np


def kalman_filter(y, F, Q, H, R, xi0, P0):
    """卡尔曼滤波。返回 (xif (T,n), Pf (T,n,n), innov (T))。"""
    T = len(y); n = F.shape[0]
    xif = np.zeros((T, n)); Pf = np.zeros((T, n, n)); innov = np.zeros(T)
    xi = xi0.copy(); P = P0.copy()
    for t in range(T):
        # 预测
        xip = F @ xi
        Pp = F @ P @ F.T + Q
        # 更新
        S = (H @ Pp @ H.T + R).item()
        eta = y[t] - (H @ xip).item()
        K = Pp @ H.T / S
        xi = xip + K * eta
        P = Pp - np.outer(K, H @ Pp)
        xif[t] = xi.ravel(); Pf[t] = P; innov[t] = eta
    return xif, Pf, innov


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi, q, r = 0.8, 1.0, 0.5
    F = np.array([[phi]]); Q = np.array([[q]])
    H = np.array([[1.0]]); R = np.array([[r]]); n = 1
    T = 400
    xi_true = np.zeros(T); y = np.zeros(T)
    for t in range(T):
        if t > 0:
            xi_true[t] = phi * xi_true[t - 1] + rng.normal(0, np.sqrt(q))
        y[t] = xi_true[t] + rng.normal(0, np.sqrt(r))

    xif, Pf, innov = kalman_filter(y, F, Q, H, R,
                                   np.zeros((n, 1)), np.eye(n) * 10.0)
    mse = np.mean((xif[:, 0] - xi_true) ** 2)
    print("滤波 MSE:", round(mse, 3), "  观测噪声方差 r =", r,
          " (滤波更优)")
    print("观测自身(y 即 x̂) MSE:", round(np.mean((y - xi_true) ** 2), 3))
    print("末态误差协方差 P:", np.round(Pf[-1], 4))
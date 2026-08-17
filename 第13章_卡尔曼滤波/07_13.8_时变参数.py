# 07_13.8 时变参数

import numpy as np


def tvp_kalman(y, x, q, r, P0=1.0):
    """时变系数回归的卡尔曼滤波, 返回 beta_t|t。"""
    T = len(y)
    beta = np.zeros(T); P = P0
    b = 0.0
    for t in range(T):
        xip = b; Pp = P + q                     # F=1, Q=q
        H = x[t]
        S = H ** 2 * Pp + r
        K = Pp * H / S
        b = xip + K * (y[t] - H * xip)
        P = (Pp - K * H * Pp)
        beta[t] = b
    return beta


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 300
    t = np.arange(T)
    beta_true = 0.5 + 0.002 * t               # 平滑漂移
    x = rng.normal(size=T)
    r = 1.0
    y = x * beta_true + rng.normal(0, np.sqrt(r), T)

    q = 0.001
    beta_hat = tvp_kalman(y, x, q, r)
    rmse = np.sqrt(np.mean((beta_hat - beta_true) ** 2))
    print("时变参数估计 RMSE =", round(rmse, 4))
    print("滤波末端 β_hat =", round(beta_hat[-1], 3), " 真值 =", round(beta_true[-1], 3))
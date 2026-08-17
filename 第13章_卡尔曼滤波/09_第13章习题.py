# 09_第13章习题

import numpy as np


def kalman_y2(y, phi, q, r):
    # 用于习题1/3: AR(1)+noise 滤波
    T = len(y)
    xi = 0.0; P = 10.0
    xif = np.zeros(T)
    for t in range(T):
        xip = phi * xi; Pp = phi ** 2 * P + q
        S = Pp + r; K = Pp / S
        xi = xip + K * (y[t] - xip)
        P = Pp - K * Pp
        xif[t] = xi
    return xif


if __name__ == "__main__":
    # 习题2: 更新方差
    Pp, R = 2.0, 0.5
    Ptt = Pp - Pp ** 2 / (Pp + R)
    print(f"习题2: 预测方差={Pp}, 观测噪声={R} => 更新方差 P_tt={Ptt:.4f}")

    # 习题1: AR(1) 无观测噪声
    print(f"习题1: F=[0.7], Q=[1.0], H=[1], R=[0]")

    # 习题3: 平滑 vs 滤波 MSE (短模拟)
    rng = np.random.default_rng(3)
    phi, q, r = 0.8, 1.0, 0.0      # 用 R 小情形(近 AR(1))
    T = 200
    xit = np.zeros(T); y = np.zeros(T)
    for t in range(T):
        if t > 0:
            xit[t] = phi * xit[t - 1] + rng.normal(0, np.sqrt(q))
        y[t] = xit[t]
    print("习题3: R=0 时滤波=精确状态(RMSE~0):",
          round(np.sqrt(np.mean((kalman_y2(y, phi, q, 0.0) - xit) ** 2)), 6))
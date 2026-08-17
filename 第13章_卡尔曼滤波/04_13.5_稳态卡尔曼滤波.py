# 04_13.5 稳态卡尔曼滤波

import numpy as np


if __name__ == "__main__":
    phi, q, r = 0.8, 1.0, 0.5
    F = np.array([[phi]]); Q = np.array([[q]]); H = np.array([[1.0]]); R = np.array([[r]])
    # 迭代 Riccati (预测形式): P_{t} = F P_{t-1}F' + Q - F P_{t-1}H'(S)^{-1}H P_{t-1}F'
    P = 10.0; vals = []
    for _ in range(100):
        S = P + r
        P = phi ** 2 * P + q - phi ** 2 * P ** 2 / S
        vals.append(P)
    print("P 收敛值:", round(vals[-1], 4), "  (前几期):",
          np.round(vals[:5], 3))
    Ks = phi * vals[-1] / (vals[-1] + r)
    print("稳态 K = FP/(HPH'+R) 形式 ≈", round(Ks, 4))
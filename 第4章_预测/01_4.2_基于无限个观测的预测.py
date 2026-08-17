# 01_4.2 基于无限个观测的预测

"""
Lecture: /第4章 预测
Content: 01_4.2 基于无限个观测的预测
"""

import numpy as np


def simulate_ar1(phi: float, sigma: float, n: int, rng) -> np.ndarray:
    """AR(1): x_t = phi x_{t-1} + eps_t (预热后返回长度 n)。"""
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    prev = 0.0
    for t in range(total):
        prev = phi * prev + eps[t]
        x[t] = prev
    return x[burn:]


def theory_mse(phi: float, sigma: float, hmax: int) -> np.ndarray:
    """h 步 MSE = sigma^2 (1 - phi^{2h})/(1 - phi^2)。"""
    h = np.arange(1, hmax + 1)
    return sigma ** 2 * (1 - phi ** (2 * h)) / (1 - phi ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi = 0.7
    sigma = 1.0
    n = 5000
    hmax = 6

    # 蒙特卡洛: 对每一条路径, 用历史端点 x_T 做 h 步预测, 累加平方误差
    B = 100
    sqerr = np.zeros((B, hmax))
    fc = np.zeros(hmax)
    for b in range(B):
        x = simulate_ar1(phi, sigma, n + hmax, rng)
        xT = x[n - 1]
        for h in range(1, hmax + 1):
            p = phi ** h * xT
            if b == 0:
                fc[h - 1] = p
            sqerr[b, h - 1] = (x[n - 1 + h] - p) ** 2

    print("h 步预测值 (h=1..6):", np.round(fc, 3), " (随 h 收敛到 0)")
    print("经验 MSE :", np.round(sqerr.mean(axis=0), 4))
    print("理论 MSE :", np.round(theory_mse(phi, sigma, hmax), 4), " (逼近 Var=3.92)")
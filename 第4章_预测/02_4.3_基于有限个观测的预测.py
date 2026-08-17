# 02_4.3 基于有限个观测的预测

"""
Lecture: /第4章 预测
Content: 02_4.3 基于有限个观测的预测
"""

import numpy as np


def simulate_ma1(theta: float, sigma: float, n: int, rng) -> tuple:
    """MA(1) x_t = eps_t + theta eps_{t-1}, 返回 (x, eps)。"""
    burn = 200
    total = n + burn
    epsv = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    pe = 0.0
    for t in range(total):
        x[t] = epsv[t] + theta * pe
        pe = epsv[t]
    return x[burn:], epsv[burn:]


def recover_innovations(theta: float, x: np.ndarray) -> np.ndarray:
    """由 MA(1) 递归还原新息: eps_hat_t = x_t - theta eps_hat_{t-1} (eps_0=0)。"""
    n = len(x)
    e = np.zeros(n)
    for t in range(n):
        e[t] = x[t] - theta * (e[t - 1] if t > 0 else 0.0)
    return e


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    theta = 0.6
    sigma = 1.0
    n = 2000
    x, eps = simulate_ma1(theta, sigma, n, rng)

    ehat = recover_innovations(theta, x)
    print("还原新息 vs 真值 |差| (末端):", np.round(np.abs(ehat[-5:] - eps[-5:]), 4),
          " (随 T 增长变小)")

    # 有限样本预测 x_{T+1} = theta * eps_T
    fc = theta * ehat[-1]
    print("1 步预测 x_{T+1} =", round(fc, 4), " (利用历史, ≈ theta*eps_T)")

    # 对比: 用历史预测 vs 忽略历史(=0)
    B = 400
    mse_hist, mse_naive = 0.0, 0.0
    for b in range(B):
        xb, epsb = simulate_ma1(theta, sigma, n + 1, rng)
        eb = recover_innovations(theta, xb[:n])
        actual = xb[n]
        mse_hist += (actual - theta * eb[-1]) ** 2
        mse_naive += actual ** 2
    print(f"有限样本预测 MSE = {mse_hist/B:.4f}  vs  忽略历史 MSE = {mse_naive/B:.4f}")
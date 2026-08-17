# 09_附录_4.B_一阶移动平均过程协方差矩阵的三角分解

"""
Lecture: /第4章 预测
Content: 09_附录_4.B_一阶移动平均过程协方差矩阵的三角分解
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def ma1_cov(theta: float, n: int) -> np.ndarray:
    """MA(1) 协方差 Toeplitz: gamma0=1+theta^2, gamma1=theta。"""
    g0 = 1 + theta ** 2
    g1 = theta
    idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    G = np.where(idx == 0, g0, np.where(idx == 1, g1, 0.0))
    return G


def recover_recursive(theta: float, x: np.ndarray) -> np.ndarray:
    """递归新息: eps_hat_t = x_t - theta eps_hat_{t-1} (eps_0=0)。"""
    n = len(x)
    e = np.zeros(n)
    for t in range(n):
        e[t] = x[t] - theta * (e[t - 1] if t > 0 else 0.0)
    return e


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    theta = 0.6
    n = 2000

    G = ma1_cov(theta, n)
    L = np.linalg.cholesky(G)
    print("||LL' - Γ|| =", np.max(np.abs(L @ L.T - G)))

    # 用 L 造 MA(1) 观测: x = L v, v ~ 白
    v = rng.normal(size=n)
    x = L @ v
    # v_hat = L^{-1} x 应≈ v
    vhat = np.linalg.solve(L, x)
    print("L^{-1}x = 新息 v, 与真 v 最大差:", np.max(np.abs(vhat - v)))

    # 三角新息 vs 递归还原新息一致
    rec = recover_recursive(theta, x)
    print("三角新息末项:", round(float(vhat[-1]), 4),
          "  递归新息末项:", round(float(rec[-1]), 4))
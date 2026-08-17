# 01_12.2 向量自回归的贝叶斯分析

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def bvar_posterior(y, X, b0, Vinv, sig2):
    """贝叶斯回归后验均值: (Vinv + X'X)/sig2 解。"""
    A = Vinv + X.T @ X / sig2
    c = Vinv @ b0 + X.T @ y / sig2
    return np.linalg.solve(A, c)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 100
    # y = 0.5*y_{t-1} + e (自己滞后系数真 0.5)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.normal()
    X = np.column_stack([np.ones(T - 1), y[:-1]])   # [常数, y_{t-1}]
    yl = y[1:]
    sig2 = 1.0

    b_ols = np.linalg.lstsq(X, yl, rcond=None)[0]
    # Minnesota 式先验: 常数→0, 一阶滞后→1, 小方差
    b0 = np.array([0.0, 1.0])
    V = np.diag([100.0, 0.5])      # 常数放松, 滞后适度收缩
    b_post = bvar_posterior(yl, X, b0, np.linalg.inv(V), sig2)

    print("OLS 系数   :", np.round(b_ols, 3))
    print("后验(收缩):", np.round(b_post, 3))
    print("先验均值   :", b0)
    print("(后验把 OLS 向 b0 收缩)")
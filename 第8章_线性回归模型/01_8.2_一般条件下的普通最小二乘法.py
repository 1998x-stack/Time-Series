# 01_8.2 一般条件下的普通最小二乘法

"""
Lecture: /第8章 线性回归模型
Content: 01_8.2 一般条件下的普通最小二乘法
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def ols_fit(y, X):
    """OLS 返回 (beta, white SE, classic SE)。"""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    T = len(y)
    k = X.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    # White 稳健: meat = sum_t e_t^2 x_t x_t'
    meat = (X * resid[:, None] ** 2).T @ X
    Vw = XtX_inv @ meat @ XtX_inv
    # 经典
    sig2 = np.sum(resid ** 2) / (T - k)
    Vc = sig2 * XtX_inv
    return beta, np.sqrt(np.diag(Vw)), np.sqrt(np.diag(Vc))


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    X = np.column_stack([np.ones(T), rng.normal(size=T), rng.normal(size=T)])
    true_beta = np.array([1.0, 2.0, -0.5])
    h = 1.0 + 2.0 * X[:, 2] ** 2      # 异方差
    B = 2000
    betas = np.zeros((B, 3))
    seeds = None
    for b in range(B):
        eps = rng.normal(0.0, 1.0, T) * np.sqrt(h)
        y = X @ true_beta + eps
        bhat, sew, sec = ols_fit(y, X)
        betas[b] = bhat
        if b == 0:
            sew0, sec0 = sew, sec

    mc_se = betas.std(axis=0)
    print("MC SE (真实) :", np.round(mc_se, 4))
    print("White SE     :", np.round(sew0, 4))
    print("经典 SE      :", np.round(sec0, 4))
    print("(White 较经典更接近 MC)")
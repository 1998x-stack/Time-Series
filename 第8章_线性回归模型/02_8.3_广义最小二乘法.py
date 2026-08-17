# 02_8.3 广义最小二乘

"""
Lecture: /第8章 线性回归模型
Content: 02_8.3 广义最小二乘
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def ar1_cov(phi, sigma2, T):
    """AR(1) 误差协方差 Omega[i,j] = sigma2*phi^{|i-j|}/(1-phi^2)。"""
    idx = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    return (sigma2 / (1 - phi ** 2)) * phi ** idx


def ols(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def gls(y, X, Omi):
    return np.linalg.inv(X.T @ Omi @ X) @ (X.T @ Omi @ y)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 120
    X = np.column_stack([np.ones(T), rng.normal(size=T), rng.normal(size=T)])
    true_beta = np.array([1.0, 0.5, -1.0])
    phi, sig2 = 0.6, 1.0
    Omega = ar1_cov(phi, sig2, T)
    Omi = np.linalg.inv(Omega)
    L = np.linalg.cholesky(Omega)

    B = 3000
    b_ols = np.zeros((B, 3)); b_gls = np.zeros((B, 3))
    for b in range(B):
        eps = L @ rng.normal(size=T)
        Y = X @ true_beta + eps
        b_ols[b] = ols(Y, X)
        b_gls[b] = gls(Y, X, Omi)

    print("OLS 经验 SE:", np.round(b_ols.std(0), 4))
    print("GLS 经验 SE:", np.round(b_gls.std(0), 4), " (应更小/更有效)")
    print("效率比 OLS/GLS SE:", np.round(b_ols.std(0) / b_gls.std(0), 3))
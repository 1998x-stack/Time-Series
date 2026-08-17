# 08_附录_4.A_普通最小二乘回归与线性投影

"""
Lecture: /第4章 预测
Content: 08_附录_4.A_普通最小二乘回归与线性投影
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def ols(y, X):
    """OLS = beta=(X'X)^{-1}X'y, 返回 (beta, fitted, resid)。"""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted
    return beta, fitted, resid


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 2000
    x1 = rng.normal(size=n)
    x2 = 0.3 * x1 + rng.normal(size=n)
    eps = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + eps

    X = np.column_stack([np.ones(n), x1, x2])
    beta, fitted, resid = ols(y, X)
    print("真系数 [截距,x1,x2]:", np.round(beta, 4), " (真值 [1,2,-0.5])")

    orth = X.T @ resid
    print("X'e (残差与各回归元正交):", np.round(orth, 8))
    print("残差·拟合值:", round(float(np.dot(resid, fitted)), 8), " (≈0)")
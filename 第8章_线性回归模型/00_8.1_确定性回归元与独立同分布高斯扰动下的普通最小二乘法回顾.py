# 00_8.1 确定性回归元与独立同分布高斯扰动下的普通最小二乘法回顾

"""
Lecture: /第8章 线性回归模型
Content: 00_8.1 确定性回归元与独立同分布高斯扰动下的普通最小二乘法回顾
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T, k = 500, 3
    X = np.column_stack([np.ones(T), rng.normal(size=T), rng.normal(size=T)])
    true_beta = np.array([1.0, 2.0, -0.5])
    sig = 1.0
    eps = rng.normal(0.0, sig, T)
    y = X @ true_beta + eps

    # OLS
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_hat
    sig_hat2 = np.sum(resid ** 2) / (T - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sig_hat2 * np.diag(XtX_inv))
    t_vals = beta_hat / se
    SSR = np.sum((X @ beta_hat - y.mean()) ** 2)
    SST = np.sum((y - y.mean()) ** 2)
    R2 = SSR / SST

    print("OLS beta =", np.round(beta_hat, 4), " (真", true_beta, ")")
    print("标准误 =", np.round(se, 4))
    print("SE 理论 =", np.round(np.sqrt(sig ** 2 * np.diag(XtX_inv)), 4))
    print("残差方差 =", round(sig_hat2, 4), " (≈1)")
    print("R² =", round(R2, 4), "  t =", np.round(t_vals, 3))
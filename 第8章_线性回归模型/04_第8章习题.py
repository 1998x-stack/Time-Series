# 04_第8章习题

"""
Lecture: /第8章 线性回归模型
Content: 04_第8章习题
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def ols_fit(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return beta, resid


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    true_beta = np.array([0.5, 1.5])

    # 习题1: 解析解
    y = X @ true_beta + rng.normal(size=T)
    beta, resid = ols_fit(y, X)
    R2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"习题1: beta_hat={np.round(beta,3)} (真{true_beta}), R²={R2:.3f}")

    # 习题2: 经典 vs White SE (异方差)
    h = 1 + 3 * X[:, 1] ** 2
    eps = rng.normal(size=T) * np.sqrt(h)
    y2 = X @ true_beta + eps
    b2, r2 = ols_fit(y2, X)
    k = X.shape[1]
    Xi = np.linalg.inv(X.T @ X)
    sig2 = np.sum(r2 ** 2) / (T - k)
    se_classic = np.sqrt(sig2 * np.diag(Xi))
    meat = (X * r2[:, None] ** 2).T @ X
    se_white = np.sqrt(np.diag(Xi @ meat @ Xi))
    print(f"习题2: 经典SE={np.round(se_classic,4)}  WhiteSE={np.round(se_white,4)}")

    # 习题3: 说明 GLS 更高效 (见 8.3 数值)
    print("习题3: 已知 Ω 时用 GLS(以 Ω⁻¹ 加权)得 BLUE")
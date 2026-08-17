# 00_4.1 预测的原理

"""
Lecture: /第4章 预测
Content: 00_4.1 预测的原理
"""

import numpy as np


def linear_projection(Y: np.ndarray, X: np.ndarray):
    """Y 对 X 的线性投影: beta = Var(X)^{-1} Cov(X,Y), alpha = mu_y - beta mu_x。"""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta_ols = np.linalg.lstsq(X1, Y, rcond=None)[0]
    pred = X1 @ beta_ols
    resid = Y - pred
    return beta_ols, pred, resid


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 20000
    mu = np.array([1.0, 2.0])          # E(X), E(Y)
    cov_xy = 0.8
    Sigma = np.array([[1.0, cov_xy], [cov_xy, 1.0]])  # VarX=1, VarY=1
    Z = rng.multivariate_normal(mu, Sigma, n)
    X, Y = Z[:, 0], Z[:, 1]

    beta_ols, pred, resid = linear_projection(Y, X)
    print("线性投影系数: alpha=%.4f beta=%.4f (真值 0.8)" % (beta_ols[0], beta_ols[1]))
    print("剩余与 X 正交(近 0):", round(float(np.dot(resid, X - X.mean())) / n, 4))

    mse_cond = np.mean(resid ** 2)
    mse_mean = np.mean((Y - Y.mean()) ** 2)
    print(f"条件 MSE = {mse_cond:.4f}  (理论 Var(Y|X)=1-rho^2={1 - cov_xy**2:.4f})")
    print(f"均值 MSE = {mse_mean:.4f}")
    print(f"MSE 之比 = {mse_cond / mse_mean:.4f} (<1, 预测优于均值)")
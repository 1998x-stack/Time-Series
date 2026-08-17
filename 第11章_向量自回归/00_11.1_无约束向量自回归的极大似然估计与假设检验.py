# 00_11.1 无约束向量自回归的极大似然估计与假设检验

# 拟合 VAR(1)(每方程 OLS = MLE)

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def fit_var(Y):
    """Y: (T, n); 逐方程 OLS y_t ~ [1, y_{t-1}]。返回 (Phi, c, Omega)。"""
    T, n = Y.shape
    X = np.column_stack([np.ones(T - 1), Y[:-1]])
    resid = np.zeros((T - 1, n))
    coeffs = np.zeros((n, n + 1))
    for j in range(n):
        b = np.linalg.lstsq(X, Y[1:, j], rcond=None)[0]
        coeffs[j] = b
        resid[:, j] = Y[1:, j] - X @ b
    c = coeffs[:, 0]
    Phi = coeffs[:, 1:]
    Omega = resid.T @ resid / (T - 1)
    return Phi, c, Omega


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi_true = np.array([[0.5, 0.1], [0.2, 0.4]])
    n = 2
    T = 5000
    L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]]))
    burn = 200; total = T + burn
    Y = np.zeros((total, n))
    for t in range(1, total):
        Y[t] = Phi_true @ Y[t - 1] + L @ rng.normal(size=n)
    Y = Y[burn:]

    Phi_hat, c_hat, Omega_hat = fit_var(Y)
    print("估计 Phi:\n", np.round(Phi_hat, 3))
    print("真值   Phi:\n", Phi_true)
    print("估计 Omega:\n", np.round(Omega_hat, 3))
# 08_附录11.B 解析导数的计算

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def anal_grad_hess(y, X, beta, sig2):
    """负对数似然(-ell=||y-Xb||^2/(2s2)) 的解析梯度与 Hessian(忽略常数)。"""
    g = -X.T @ (y - X @ beta) / sig2
    H = X.T @ X / sig2
    return g, H


def fd_grad(f, beta, h=1e-6):
    grad = np.zeros_like(beta)
    for i in range(len(beta)):
        bp, bm = beta.copy(), beta.copy()
        bp[i] += h; bm[i] -= h
        grad[i] = (f(bp) - f(bm)) / (2 * h)
    return grad


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T, k = 500, 3
    X = np.column_stack([np.ones(T), rng.normal(size=T), rng.normal(size=T)])
    beta_true = np.array([1.0, 2.0, -0.5])
    sig2 = 1.0
    y = X @ beta_true + rng.normal(scale=np.sqrt(sig2), size=T)
    beta0 = np.array([0.0, 0.0, 0.0])

    def negll(b):
        return np.sum((y - X @ b) ** 2) / (2 * sig2)

    g_an = anal_grad_hess(y, X, beta0, sig2)[0]
    H_an = anal_grad_hess(y, X, beta0, sig2)[1]
    g_fd = fd_grad(negll, beta0)
    print("解析梯度:", np.round(g_an, 4))
    print("数值梯度:", np.round(g_fd, 4))
    print("梯度最大差:", np.round(np.max(np.abs(g_an - g_fd)), 12))
    print("解析Hessian[0,0] =", round(H_an[0, 0], 4), " = 理论 T/sig2 =", round(T / sig2, 4))
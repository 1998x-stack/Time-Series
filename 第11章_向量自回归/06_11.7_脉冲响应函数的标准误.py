# 06_11.7 脉冲响应函数的标准误

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def fit_var(Y):
    n = Y.shape[1]
    X = np.column_stack([np.ones(len(Y) - 1), Y[:-1]])
    coeffs = np.zeros((n, n + 1))
    for j in range(n):
        coeffs[j] = np.linalg.lstsq(X, Y[1:, j], rcond=None)[0]
    return coeffs[:, 1:], coeffs[:, 0]


def irf_element(Phi, h, j=0, i=0):
    """冲击 j -> 变量 i, 在 horizon h 的脉冲响应。"""
    n = Phi.shape[0]
    z = np.zeros(n); z[j] = 1.0
    for _ in range(h):
        z = Phi @ z
    return z[i]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi_true = np.array([[0.5, 0.2], [0.1, 0.4]])
    L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]]))
    T = 4000
    burn = 100; total = T + burn
    Y = np.zeros((total, 2))
    for t in range(1, total):
        Y[t] = Phi_true @ Y[t - 1] + L @ rng.normal(size=2)
    Y = Y[burn:]

    Phi_hat, _ = fit_var(Y)
    resid = Y[1:] - (Y[:-1] @ Phi_hat.T)
    i, j, h = 1, 0, 3                      # 冲击0 -> 变量1, 3期后
    point = irf_element(Phi_hat, h, j, i)

    B = 400
    irfs = np.zeros(B)
    for b in range(B):
        rb = resid[rng.integers(0, len(resid), size=len(resid))]
        Yb = np.zeros((len(resid) + 1, 2))
        for t in range(1, len(resid) + 1):
            Yb[t] = Phi_hat @ Yb[t - 1] + rb[t - 1]
        Pib, _ = fit_var(Yb[1:])
        irfs[b] = irf_element(Pib, h, j, i)

    print(f"IRF(冲击0→变量1, 3期) 点估={point:.4f}  bootstrap SE={irfs.std():.4f}")
    lo, hi = np.percentile(irfs, [2.5, 97.5])
    print(f"95% 置信区间 = [{lo:.4f}, {hi:.4f}]")
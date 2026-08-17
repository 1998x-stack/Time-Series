# 02_11.3 有约束向量自回归的极大似然估计

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def fit_var(Y):
    """无约束 VAR(1) 逐方程 OLS, 返回 Phi, c。"""
    n = Y.shape[1]
    X = np.column_stack([np.ones(len(Y) - 1), Y[:-1]])
    coeffs = np.zeros((n, n + 1))
    for j in range(n):
        coeffs[j] = np.linalg.lstsq(X, Y[1:, j], rcond=None)[0]
    return coeffs[:, 1:], coeffs[:, 0]


def logdet(Omega):
    return np.log(np.linalg.det(Omega))


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi_true = np.array([[0.5, 0.3], [0.0, 0.4]])   # 真: Phi[1,0]=0
    T = 3000
    burn = 100; total = T + burn
    Y = np.zeros((total, 2))
    for t in range(1, total):
        Y[t] = Phi_true @ Y[t - 1] + rng.normal(size=2)
    Y = Y[burn:]
    y1, y2 = Y[:, 0], Y[:, 1]

    # 无约束
    Phi_u, c_u = fit_var(Y)
    rest_u = Y[1:] - (Y[:-1] @ Phi_u.T + c_u)
    Om_u = rest_u.T @ rest_u / (T - 1)
    lda = logdet(Om_u)

    # 受限: 施加 Phi[1,0]=0 (y1 不从 y2 方程去掉待测试: 这里测试的是真约束 Phi[1,0]=0)
    # 受限时 y2 方程仅用 y2 滞后, y1 方程不变
    b2 = np.linalg.lstsq(np.column_stack([np.ones(len(y2) - 1), y2[:-1]]),
                         y2[1:], rcond=None)[0]
    resid = np.zeros((T - 1, 2))
    resid[:, 1] = y2[1:] - (b2[0] + b2[1] * y2[:-1])
    resid[:, 0] = y1[1:] - (c_u[0] + y1[:-1] * Phi_u[0, 0] + y2[:-1] * Phi_u[0, 1])
    Om_r = resid.T @ resid / (T - 1)
    ldr = logdet(Om_r)

    LR = (T - 1) * (ldr - lda)
    p = 1 - stats.chi2.cdf(LR, 1)
    print(f"无约束 log|Ω|={lda:.4f}  受限 log|Ω|={ldr:.4f}  LR={LR:.3f}  p={p:.4f}")
    print("受限约束(Phi[1,0]=0)是否被拒?", p < 0.05)
# 03_9.4 完全信息极大似然估计

"""
Lecture: /第9章 线性联立方程系统
Content: 03_9.4 完全信息极大似然估计
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def neg_loglik_fiml(par, y1, y2, x1, x2):
    alpha, b1, b21, b22 = par[:4]
    # Sigma (2x2) 用 Cholesky 参数化保证正定
    L = np.array([[np.exp(par[4]), 0.0], [par[5], np.exp(par[6])]])
    Sigma = L @ L.T
    e1 = y1 - alpha * y2 - b1 * x1
    e2 = y2 - b21 * x1 - b22 * x2
    Sinv = np.linalg.inv(Sigma)
    ll = len(y1) / 2 * np.log(np.linalg.det(Sigma))
    for i in range(len(y1)):
        v = np.array([e1[i], e2[i]])
        ll += 0.5 * v @ Sinv @ v
    return ll if np.isfinite(ll) else 1e18


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 1000
    true = dict(alpha=0.8, b1=0.5, b21=1.0, b22=0.7)
    x1 = rng.normal(size=T); x2 = rng.normal(size=T)
    e2 = rng.normal(size=T); e1 = 0.6 * e2 + np.sqrt(1-0.36) * rng.normal(size=T)
    y2 = true["b21"] * x1 + true["b22"] * x2 + e2
    y1 = true["alpha"] * y2 + true["b1"] * x1 + e1

    res = minimize(lambda p: neg_loglik_fiml(p, y1, y2, x1, x2),
                   x0=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   method="Nelder-Mead", options=dict(maxiter=60000, xatol=1e-8, fatol=1e-10))
    a_fiml = res.x[0]
    # 对照 2SLS
    Z = np.column_stack([x1, x2])
    y2hat = Z @ np.linalg.lstsq(Z, y2, rcond=None)[0]
    a_tsls = np.linalg.lstsq(np.column_stack([y2hat, x1]), y1, rcond=None)[0][0]
    print("FIML alpha =", round(a_fiml, 4), "  2SLS alpha =", round(a_tsls, 4),
          "  真值 =", true["alpha"])
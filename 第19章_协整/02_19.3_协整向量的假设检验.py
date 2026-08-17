# 02_19.3 协整向量的假设检验

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 4000
    beta_true = 1.5
    y2 = np.cumsum(rng.normal(0, 1, T))
    y1 = beta_true * y2 + rng.normal(0, 0.5, T)
    X = np.column_stack([np.ones(T), y2])
    b = np.linalg.lstsq(X, y1, rcond=None)[0]
    e = y1 - X @ b
    s2 = np.sum(e ** 2) / (T - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    beta_hat = b[1]

    print(f"协整系数 β̂ = {beta_hat:.4f} (真 1.5), SE={se:.4f}")
    for b0 in (1.5, 2.0):
        t = (beta_hat - b0) / se
        p = 2 * (1 - stats.norm.cdf(abs(t)))
        print(f"  H0: β={b0}: t={t:.2f}, p={p:.2e} -> {'接受' if p>0.05 else '拒绝'}")
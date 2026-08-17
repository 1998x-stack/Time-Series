# 03_附录18.A 第18章性质证明

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def spurious_stats(T, rng):
    y = np.cumsum(rng.normal(0, 1, T)); x = np.cumsum(rng.normal(0, 1, T))
    X = np.column_stack([np.ones(T), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = np.sum(e ** 2) / (T - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    R2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    return abs(b[1] / se), R2


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    B = 1000
    for T in (50, 200, 1000):
        res = np.array([spurious_stats(T, rng) for _ in range(B)])
        print(f"T={T:5d}: 平均|t| = {res[:,0].mean():.1f}, 平均R² = {res[:,1].mean():.3f}")
# 01_16.2 简单时间趋势模型的假设检验

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def delta_tstat(T, rng):
    t = np.arange(T).astype(float)
    y = rng.normal(0, 1, T)             # H0: delta=0
    X = np.column_stack([np.ones(T), t])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = np.sum(e ** 2) / (T - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return b[1] / se


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 300
    B = 5000
    tt = np.array([delta_tstat(T, rng) for _ in range(B)])
    print("H0: δ=0 下 t 统计:")
    print("  均值", round(tt.mean(), 3), "  方差", round(tt.var(), 3), " (应 0/1)")
    q = np.percentile(tt, [2.5, 97.5])
    print("  2.5/97.5 分位 =", np.round(q, 3), " vs ±1.96 (标准正态)")
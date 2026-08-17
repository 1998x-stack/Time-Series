# 00_15.1 简介

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    t = np.arange(T).astype(float)

    # 趋势平稳型
    a, b = 0.5, 0.02
    ts = a + b * t + rng.normal(0, 1, T)
    X = np.column_stack([np.ones(T), t])
    resid = ts - X @ np.linalg.lstsq(X, ts, rcond=None)[0]
    print("TS 去趋势残差: 前/后100期方差", round(resid[:100].var(), 3),
          "/", round(resid[100:].var(), 3), " (平稳, 相近)")

    # 单位根: Var(y_t) 跨重复随 t 增长 (~ sigma^2 t)
    B = 2000
    for tt in (50, 100, 200):
        draws = np.array([np.sum(rng.normal(0, 1, tt)) for _ in range(B)])
        print(f"  单位根 Var(y_{{{tt}}}) ≈ {draws.var():.2f} (σ²·t={tt})")
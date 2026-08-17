# 00_15.1 简介

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    t = np.arange(T).astype(float)

    # 趋势平稳 TS: y = a + b t + e
    a, b = 0.5, 0.02
    ts = a + b * t + rng.normal(0, 1, T)
    # 单位根 RW
    rw = np.cumsum(rng.normal(0, 1, T))

    # TS 去趋势 (回归残差) 应近似平稳
    X = np.column_stack([np.ones(T), t])
    resid = ts - X @ np.linalg.lstsq(X, ts, rcond=None)[0]
    print("TS 去趋势残差: 前100期方差", round(resid[:100].var(), 3),
          " 后100期方差", round(resid[100:].var(), 3), " (平稳, 相近)")

    # 单位根: 方差随 t 增大
    print("单位根: 前100期方差", round(rw[:100].var(), 3),
          " -> 后100期方差", round(rw[100:].var(), 3), " (随 t 增长)")
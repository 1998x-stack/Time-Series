# 00_19.1 简介

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def adf(y):
    """ADF(1) t 检验残差平稳 / 是否单位根。"""
    T = len(y); dy = np.diff(y)
    X = np.column_stack([np.ones(T - 1), y[:-1]])
    b = np.linalg.lstsq(X, dy, rcond=None)[0]
    e = dy - X @ b
    s2 = np.sum(e ** 2) / (T - 3)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return b[1] / se


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 2000
    beta = 1.5
    y2 = np.cumsum(rng.normal(0, 1, T))          # I(1)
    u = rng.normal(0, 0.5, T)   # 平稳残差
    y1 = beta * y2 + u                          # I(1) 但残差 I(0)
    z = y1 - beta * y2                          # 真实残差
    print("ADF(水平 y1) =", round(adf(y1), 2), "  (不显著, I(1))")
    print("ADF(协整残差 z) =", round(adf(z), 2), " (应<<-2.86, 平稳)")
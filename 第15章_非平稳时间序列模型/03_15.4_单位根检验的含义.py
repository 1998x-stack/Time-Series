# 03_15.4 单位根检验的含义

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def df_stat(y):
    """DF t 统计: 回归 Δy_t = a + gamma y_{t-1} + u, 报告 gamma 的 t。"""
    n = len(y)
    dy = np.diff(y)
    X = np.column_stack([np.ones(n - 1), y[:-1]])
    b = np.linalg.lstsq(X, dy, rcond=None)[0]
    u = dy - X @ b
    k = 2
    s2 = np.sum(u ** 2) / (n - 1 - k)
    se_gamma = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return b[1] / se_gamma


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    B = 5000
    taus = np.zeros(B)
    for b in range(B):
        rw = np.cumsum(rng.normal(0, 1, T))
        taus[b] = df_stat(rw)
    q5 = np.percentile(taus, 5)
    print(f"DF t 的 5% 分位 = {q5:.2f}  (正态 -1.645)")
    print(f"含常数 DF 临界值 ≈ -2.86; 本模拟 {q5:.2f}")
    print("含义: 须用 DF 临界值, 普通 t 放弃 (1.65) 会高估拒绝")
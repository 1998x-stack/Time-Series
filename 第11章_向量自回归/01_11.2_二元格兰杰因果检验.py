# 01_11.2 二元格兰杰因果检验

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def granger_test(y, x, p=1):
    """检验 x 是否格兰杰导致 y (水平至 p 滞后)。

    无约束: y_t ~ [1, y_{t-1}, x_{t-1}]
    受限:   y_t ~ [1, y_{t-1}]
    返回 F 统计与 p 值。
    """
    T = len(y)
    idx = np.arange(p, T)
    yl = y[idx]
    Xr = np.column_stack([np.ones(len(idx)), y[idx - 1]])
    Xu = np.column_stack([np.ones(len(idx)), y[idx - 1], x[idx - 1]])

    def rss(X):
        b = np.linalg.lstsq(X, yl, rcond=None)[0]
        return np.sum((yl - X @ b) ** 2)

    rss_r, rss_u = rss(Xr), rss(Xu)
    T_, k_ = Xu.shape
    q = Xu.shape[1] - Xr.shape[1]
    F = ((rss_r - rss_u) / q) / (rss_u / (T_ - k_))
    p_val = 1 - stats.f.cdf(F, q, T_ - k_)
    return F, p_val


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi = np.array([[0.5, 0.3], [0.0, 0.4]])   # 仅 y2 -> y1
    n = 2; T = 4000
    L = np.eye(n)
    burn = 100; total = T + burn
    Y = np.zeros((total, n))
    for t in range(1, total):
        Y[t] = Phi @ Y[t - 1] + L @ rng.normal(size=n)
    Y = Y[burn:]
    y1, y2 = Y[:, 0], Y[:, 1]

    F12, p12 = granger_test(y1, y2)
    F21, p21 = granger_test(y2, y1)
    print(f"y2 → y1: F={F12:.2f} p={p12:.4f}  显著={p12 < 0.05}  (真: 是)")
    print(f"y1 → y2: F={F21:.2f} p={p21:.4f}  显著={p21 < 0.05}  (真: 否)")
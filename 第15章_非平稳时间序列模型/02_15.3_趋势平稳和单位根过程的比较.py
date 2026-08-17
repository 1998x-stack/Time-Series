# 02_15.3 趋势平稳和单位根过程的比较

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def acf(x, hmax=5):
    n = len(x); xc = x - x.mean()
    return [np.dot(xc[h:], xc[: n - h]) / np.dot(xc, xc) for h in range(1, hmax + 1)]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 800
    t = np.arange(T).astype(float)
    # TS: determin trend + 白噪声
    ts = 0.5 + 0.02 * t + rng.normal(0, 1, T)
    # 单位根带漂移
    ur = 0.02 * t + np.cumsum(rng.normal(0, 1, T))

    X = np.column_stack([np.ones(T), t])
    b = np.linalg.lstsq(X, ts, rcond=None)[0]
    res_ts = ts - X @ b
    b2 = np.linalg.lstsq(X, ur, rcond=None)[0]
    res_ur = ur - X @ b2  # 去趋势后的单位根残差

    print("TS 去趋势残差 ACF(1..5):", np.round(acf(res_ts), 3), " (衰减)")
    print("UR 去趋势残差 ACF(1..5):", np.round(acf(res_ur), 3), " (高, 不衰减)")
    # 差分后 UR 平稳
    dur = np.diff(ur)
    print("UR 差分残差 ACF(1..5):", np.round(acf(dur), 3), " (≈0, 平稳)")
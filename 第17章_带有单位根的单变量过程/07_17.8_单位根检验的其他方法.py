# 07_17.8 单位根检验的其他方法

import numpy as np


def acf(x, hmax=3):
    n = len(x); xc = x - x.mean()
    return [np.dot(xc[h:], xc[: n - h]) / np.dot(xc, xc) for h in range(1, hmax + 1)]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 4000
    rw = np.cumsum(rng.normal(0, 1, T))          # 单位根
    ar = np.zeros(T)
    for t in range(1, T):
        ar[t] = 0.6 * ar[t - 1] + rng.normal()    # 平稳
    print("单位根差分 Δy ACF:", np.round(acf(np.diff(rw)), 3), " (≈0, 差分恰当)")
    print("平稳序列差分 Δy ACF:", np.round(acf(np.diff(ar)), 3), " (一阶为负, 过度差分)")
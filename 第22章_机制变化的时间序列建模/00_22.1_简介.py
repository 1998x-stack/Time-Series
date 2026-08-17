# 00_22.1 简介

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 4000
    mu = np.array([1.0, -1.0]); sig = np.array([0.5, 0.5])
    P = np.array([[0.98, 0.02], [0.02, 0.98]])   # 转移(两状态)
    S = np.zeros(T, dtype=int)
    for t in range(1, T):
        S[t] = 1 - S[t - 1] if rng.random() > P[S[t - 1], S[t - 1]] else S[t - 1]
    y = mu[S] + sig[S] * rng.normal(size=T)
    for k in (0, 1):
        mask = S == k
        print(f"状态{k}: 比例={mask.mean():.2f}, 均值={y[mask].mean():.3f} (真 {mu[k]})")
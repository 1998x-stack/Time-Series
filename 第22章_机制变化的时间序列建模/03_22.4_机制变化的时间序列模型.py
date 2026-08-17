# 03_22.4 机制变化的时间序列模型

import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def stationary(P):
    A = np.vstack([P.T - np.eye(2), np.ones((1, 2))])
    b = np.array([0.0, 0.0, 1.0])
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi / pi.sum()


def hamilton_filter(y, mu, sig, P):
    """Hamilton 滤波: 返回每期状态1的后验 P(S_t=1|y)。"""
    T = len(y)
    pred = stationary(P)
    ps1 = np.zeros(T)
    for t in range(T):
        dens = norm.pdf(y[t], mu, sig)          # 各状态密度
        w = pred * dens                          # 未归一后验
        w = w / (w.sum() + 1e-12)
        ps1[t] = w[1]
        pred = w @ P                            # 递推下一期预测
    return ps1


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 1500
    mu = np.array([1.0, -1.0]); sig = np.array([0.5, 0.5])
    P = np.array([[0.97, 0.03], [0.02, 0.98]])
    S = np.zeros(T, dtype=int)
    for m in range(1, T):
        S[m] = 1 - S[m - 1] if rng.random() > P[S[m - 1], S[m - 1]] else S[m - 1]
    y = mu[S] + sig[S] * rng.normal(size=T)

    ps1 = hamilton_filter(y, mu, sig, P)
    agree = np.mean((ps1 > 0.5).astype(int) == S)
    corr = np.corrcoef(ps1, S.astype(float))[0, 1]
    print(f"Hamilton 滤波: 真实 vs 后验的 判定相符率 = {agree:.3f}, 相关 = {corr:.3f}")
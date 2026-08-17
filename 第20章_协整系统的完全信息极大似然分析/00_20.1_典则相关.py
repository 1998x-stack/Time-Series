# 00_20.1 典则相关

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")
from scipy.linalg import eigh


def johansen_eigenvalues(Y, k=1):
    """Johansen 广义特征值: |λ S11 - S10 S00^-1 S01|=0 (对称 eigh)。

    Y (T,n)。返回 λ∈[0,1](平方典则相关)。
    """
    T, n = Y.shape
    dy = np.diff(Y, axis=0)
    yl = Y[:-1]
    R0 = dy - dy.mean(0)
    R1 = yl - yl.mean(0)
    S00 = R0.T @ R0 / T; S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T; S10 = S01.T
    G = S10 @ np.linalg.inv(S00) @ S01
    ev = eigh(G, S11, eigvals_only=True)
    return np.clip(ev, 0, 1)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 4000
    # 协整系: y2 随机游走; y1 向 1.5*y2 误差修正 (有真实 ECM)
    beta = 1.5; alpha = 0.5
    y2 = np.zeros(T); y1 = np.zeros(T)
    for t in range(1, T):
        y2[t] = y2[t - 1] + rng.normal()
        z = y1[t - 1] - beta * y2[t - 1]
        y1[t] = y1[t - 1] - alpha * z + beta * (y2[t] - y2[t - 1]) + 0.3 * rng.normal()
    Yc = np.column_stack([y1, y2])
    # 不协整
    x1 = np.cumsum(rng.normal(0, 1, T)); x2 = np.cumsum(rng.normal(0, 1, T))
    Xn = np.column_stack([x1, x2])

    lc = johansen_eigenvalues(Yc)
    ln = johansen_eigenvalues(Xn)
    print("协整系统 λ:", np.round(lc, 3), " (个别显著>0)")
    print("不协整 λ:", np.round(ln, 3), " (均远离 1)")
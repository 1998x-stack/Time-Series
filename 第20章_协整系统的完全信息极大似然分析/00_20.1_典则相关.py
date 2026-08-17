# 00_20.1 典则相关

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def johansen_eigenvalues(Y, k=1):
    """Johansen 广义特征值: S00^{-1} S01 S11^{-1} S10。Y (T,n)。"""
    T, n = Y.shape
    dy = np.diff(Y, axis=0)
    yl = Y[:-1]
    R0 = dy - dy.mean(0)
    R1 = yl - yl.mean(0)
    S00 = R0.T @ R0 / T; S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T; S10 = S01.T
    M = np.linalg.inv(S00) @ S01 @ np.linalg.inv(S11) @ S10
    return np.linalg.eigvalsh(M)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 3000
    # 协整: y2=随机游走, y1=1.5 y2 + 平稳
    y2 = np.cumsum(rng.normal(0, 1, T))
    y1 = 1.5 * y2 + rng.normal(0, 0.5, T)
    Yc = np.column_stack([y1, y2])
    # 不协整: 两个独立随机游走
    x1 = np.cumsum(rng.normal(0, 1, T)); x2 = np.cumsum(rng.normal(0, 1, T))
    Xn = np.column_stack([x1, x2])

    lc = johansen_eigenvalues(Yc)
    ln = johansen_eigenvalues(Xn)
    print("协整系统 λ:", np.round(lc, 3), " (一个≈1)")
    print("不协整 λ:", np.round(ln, 3), " (均远离 1)")
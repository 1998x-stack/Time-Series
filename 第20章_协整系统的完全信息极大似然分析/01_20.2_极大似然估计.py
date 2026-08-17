# 01_20.2 极大似然估计

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")
from scipy.linalg import eigh


def johansen_beta(Y, k=1):
    """Johansen 协整向量(最大特征值对应特征向量)。"""
    T, n = Y.shape
    dy = np.diff(Y, axis=0); yl = Y[:-1]
    R0 = dy - dy.mean(0); R1 = yl - yl.mean(0)
    S00 = R0.T @ R0 / T; S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T; S10 = S01.T
    G = S10 @ np.linalg.inv(S00) @ S01
    w, v = eigh(G, S11)          # 特征向量 (列)
    idx = np.argsort(w)          # 升序
    beta = v[:, idx[-1]]         # 最大特征值对应向量
    return beta / beta[0]        # 归一(第一个分量为1)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 6000
    beta = 1.5; alpha = 0.5
    y2 = np.zeros(T); y1 = np.zeros(T)
    for t in range(1, T):
        y2[t] = y2[t - 1] + rng.normal()
        z = y1[t - 1] - beta * y2[t - 1]
        y1[t] = y1[t - 1] - alpha * z + beta * (y2[t] - y2[t - 1]) + 0.3 * rng.normal()
    Y = np.column_stack([y1, y2])
    b = johansen_beta(Y)
    print("Johansen β̂(归一) =", np.round(b, 3), "  真向量 = [1,", -beta, "]")
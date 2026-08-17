# 02_20.3 假设检验

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")
from scipy.linalg import eigh


def johansen_lambdas(Y):
    T, n = Y.shape
    dy = np.diff(Y, axis=0); yl = Y[:-1]
    R0 = dy - dy.mean(0); R1 = yl - yl.mean(0)
    S00 = R0.T @ R0 / T; S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T; S10 = S01.T
    G = S10 @ np.linalg.inv(S00) @ S01
    return np.clip(eigh(G, S11, eigvals_only=True), 0, 1), T


def trace_stat(lamb, T, r):
    l = np.sort(lamb)[::-1]
    return -T * np.sum(np.log(1 - l[r:]))


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 2000
    # 协整
    beta = 1.5; alpha = 0.5
    y2 = np.zeros(T); y1 = np.zeros(T)
    for t in range(1, T):
        y2[t] = y2[t - 1] + rng.normal()
        z = y1[t - 1] - beta * y2[t - 1]
        y1[t] = y1[t - 1] - alpha * z + beta * (y2[t] - y2[t - 1]) + 0.3 * rng.normal()
    Lamb_co, _ = johansen_lambdas(np.column_stack([y1, y2]))
    tr_co = trace_stat(Lamb_co, T, 0)

    # 不协整
    x1 = np.cumsum(rng.normal(0, 1, T)); x2 = np.cumsum(rng.normal(0, 1, T))
    Lamb_no, _ = johansen_lambdas(np.column_stack([x1, x2]))
    tr_no = trace_stat(Lamb_no, T, 0)
    print(f"协整   H0:秩0 迹 = {tr_co:.1f} (大, 拒绝无协整)")
    print(f"不协整 H0:秩0 迹 = {tr_no:.1f} (小, 接受无协整)")
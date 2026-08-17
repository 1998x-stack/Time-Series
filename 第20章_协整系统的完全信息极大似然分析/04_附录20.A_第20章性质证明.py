# 04_附录20.A 第20章性质证明

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")
from scipy.linalg import eigh


def max_lambda(Y):
    T, n = Y.shape
    dy = np.diff(Y, axis=0); yl = Y[:-1]
    R0 = dy - dy.mean(0); R1 = yl - yl.mean(0)
    S00 = R0.T @ R0 / T; S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T; S10 = S01.T
    G = S10 @ np.linalg.inv(S00) @ S01
    w = np.clip(eigh(G, S11, eigvals_only=True), 0, 1)
    return w.max()


def coint_data(T, rng):
    beta = 1.5; alpha = 0.5
    y2 = np.zeros(T); y1 = np.zeros(T)
    for t in range(1, T):
        y2[t] = y2[t - 1] + rng.normal()
        z = y1[t - 1] - beta * y2[t - 1]
        y1[t] = y1[t - 1] - alpha * z + beta * (y2[t] - y2[t - 1]) + 0.3 * rng.normal()
    return np.column_stack([y1, y2])


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    B = 150
    print("协整最大λ均值     不协整最大λ均值")
    for T in (200, 800, 3200):
        co = np.mean([max_lambda(coint_data(T, rng)) for _ in range(min(B, 150))])
        no = np.mean([max_lambda(np.column_stack([np.cumsum(rng.normal(0,1,T)),
                                                  np.cumsum(rng.normal(0,1,T))])) for _ in range(min(B,150))])
        print(f"T={T:5d}:   {co:.4f}          {no:.5f}")
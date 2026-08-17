# 05_第20章习题

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")
from scipy.linalg import eigh


def johansen(Y):
    T, n = Y.shape
    dy = np.diff(Y, axis=0); yl = Y[:-1]
    R0 = dy - dy.mean(0); R1 = yl - yl.mean(0)
    S00 = R0.T @ R0 / T; S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T; S10 = S01.T
    G = S10 @ np.linalg.inv(S00) @ S01
    w, v = eigh(G, S11)
    idx = np.argsort(w)
    beta = v[:, idx[-1]]
    w = np.clip(w, 0, 1)
    tr = -T * np.sum(np.log(1 - np.clip(w, 0, 1 - 1e-9)))
    return w, beta, tr


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 2000
    beta = 1.5; alpha = 0.5
    y2 = np.zeros(T); y1 = np.zeros(T)
    for t in range(1, T):
        y2[t] = y2[t - 1] + rng.normal()
        z = y1[t - 1] - beta * y2[t - 1]
        y1[t] = y1[t - 1] - alpha * z + beta * (y2[t] - y2[t - 1]) + 0.3 * rng.normal()
    Y = np.column_stack([y1, y2])
    w, b, tr = johansen(Y)
    print("习题1/3: λ =", np.round(w, 3))
    print("习题2:   协整向量 β̂ =", np.round(b / b[0], 3), "(真", beta, ")")
    print("习题:    秩0 迹 =", round(tr, 1), "(大→拒绝无协整)")
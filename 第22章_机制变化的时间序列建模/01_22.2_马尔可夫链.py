# 01_22.2 马尔可夫链

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def stationary(P):
    """平稳分布 π: 解 πP=π 使 π·1=1。"""
    A = np.vstack([(P.T - np.eye(2)), np.ones((1, 2))])  # (3,2)
    b = np.array([0.0, 0.0, 1.0])
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi / pi.sum()


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    P = np.array([[0.97, 0.03], [0.01, 0.99]])
    T = 100000
    S = np.zeros(T, dtype=int)
    for t in range(1, T):
        u = rng.random()
        S[t] = 0 if u < P[S[t - 1], 0] else 1

    Pest = np.zeros((2, 2))
    for i in range(2):
        nxt = S[1:][S[:-1] == i]
        Pest[i] = [np.mean(nxt == 0), np.mean(nxt == 1)]
    pi_th = stationary(P)
    emp = np.array([np.mean(S == 0), np.mean(S == 1)])
    print("估计转移矩阵 P̂:\n", np.round(Pest, 3))
    print("平稳分布 π 理论:", np.round(pi_th, 3), "  经验:", np.round(emp, 3))
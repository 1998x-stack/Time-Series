# 09_第11章习题

import numpy as np


if __name__ == "__main__":
    # 习题3: 长期响应 = (I-Phi)^{-1} = 累积脉冲响应之和
    Phi = np.array([[0.5, 0.1], [0.2, 0.3]])
    n = 2
    long_run = np.linalg.inv(np.eye(n) - Phi)
    # 累加响应: sum_k Psi_k = sum_k Phi^k
    cum = np.zeros((n, n)); A = np.eye(n)
    for _ in range(1000):
        cum += A
        A = A @ Phi
    print("习题3: 长期响应 (I-Φ)^{-1}:\n", np.round(long_run, 4))
    print("       累积 sum_k Ψ_k:\n", np.round(cum, 4))
    # 习题1: 平稳性
    ev = np.linalg.eigvals(Phi)
    print("习题1: 特征值模", np.round(np.abs(ev), 4), "平稳:", bool(np.max(np.abs(ev)) < 1))
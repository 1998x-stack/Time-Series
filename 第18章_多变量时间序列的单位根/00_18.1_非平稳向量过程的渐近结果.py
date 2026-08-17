# 00_18.1 非平稳向量过程的渐近结果

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Omega = np.array([[1.0, 0.5], [0.5, 1.0]])
    L = np.linalg.cholesky(Omega)
    T = 20000
    B = 2000
    # 向量缩放: W_T(s=1) = (1/sqrt T) sum v_t -> N(0, Omega)
    W = np.zeros((B, 2))
    for b in range(B):
        v = L @ rng.normal(size=(T, 2)).T  # (2,T)
        W[b] = v.sum(axis=1) / np.sqrt(T)
    print("缩放向量 W(1) 的协方差:")
    print(np.round(np.cov(W.T), 3))
    print("渐近应为 Ω:\n", Omega)
# 01_18.2 包含单位根的向量自回归过程

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    Phi = np.array([[1.0, 0.0], [0.0, 0.5]])   # 特征值 1 与 0.5
    ev = np.linalg.eigvals(Phi)
    print("Φ 特征值:", np.round(ev, 3), " (含 1 => 单位根)")

    T = 2000
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = Phi @ Y[t - 1] + rng.normal(size=2) * 0.5 + np.array([rng.normal(), 0.0])
    Y[0] = [0.0, 0.0]
    print("y1(单位根): 前/后段方差 =",
          round(Y[:1000, 0].var(), 2), "/", round(Y[1000:, 0].var(), 2),
          " (增长)")
    print("y2(平稳):   前/后段方差 =",
          round(Y[:1000, 1].var(), 2), "/", round(Y[1000:, 1].var(), 2),
          " (稳定)")
# 02_14.3 拓展

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def gmm_estimate(y, x, z, W):
    Mzx = (z * x[:, None]).mean(axis=0)
    Mzy = (z * y[:, None]).mean(axis=0)
    a = float(Mzx @ W @ Mzx); b = float(Mzx @ W @ Mzy)
    return b / a


def j_stat(y, x, z, theta, W):
    m = (z * (y - x * theta)[:, None]).mean(axis=0)
    return float(m @ W @ m)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 3000
    x = rng.normal(0, 1, T)
    e = rng.normal(0, 1, T)
    y = x * 2.0 + e
    z = np.column_stack([x, x ** 2, x ** 3])      # 3 个工具 > 1 参数(过度)
    L = z.shape[1]

    W1 = np.eye(L)
    th1 = gmm_estimate(y, x, z, W1)
    u = y - x * th1
    Om = (z * u[:, None]).T @ (z * u[:, None]) / T
    W2 = np.linalg.inv(Om + 1e-8 * np.eye(L))
    th2 = gmm_estimate(y, x, z, W2)
    J = T * j_stat(y, x, z, th2, W2)
    p = 1 - stats.chi2.cdf(J, L - 1)
    print(f"两阶段 θ = {th2:.4f} (真 2.0)")
    print(f"J = {J:.3f}, 自由度 {L-1}, p = {p:.4f} (真模型应通过 J>p 0.05)")
# 05_4.6 高斯过程的最优预测

"""
Lecture: /第4章 预测
Content: 05_4.6 高斯过程的最优预测
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    n = len(x)
    xc = x - x.mean()
    return np.array([np.dot(xc[h:], xc[: n - h]) / np.dot(xc, xc)
                     for h in range(maxlag + 1)])


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi = 0.7
    sigma = 1.0
    n = 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, sigma)

    # 最优 1 步预测: phi*x_t, 残差即新息
    resid = x[1:] - phi * x[:-1]
    print("最优预测残差方差 =", round(resid.var(), 4), "≈ sigma^2 =", sigma ** 2)
    rho = sample_acf(resid, 4)
    print("残差 ACF (h=1..4) =", np.round(rho[1:], 4), " (≈0, 白噪声)")

    # 用错误系数 (0.5) 预测 -> MSE 增大
    resid2 = x[1:] - 0.5 * x[:-1]
    print(f"正确系数 MSE = {resid.var():.4f},  错误系数(0.5) MSE = {resid2.var():.4f} (更大)")
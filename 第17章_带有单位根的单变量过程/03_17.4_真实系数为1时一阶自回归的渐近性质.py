# 03_17.4 真实系数为1时一阶自回归的渐近性质

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def rho_hat_unitroot(T, rng):
    y = np.cumsum(rng.normal(0, 1, T))
    rho = np.dot(y[:-1], y[1:]) / np.dot(y[:-1], y[:-1])
    return rho


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 1000
    B = 10000
    rhos = np.array([rho_hat_unitroot(T, rng) for _ in range(B)])
    tr = T * (rhos - 1.0)
    print(f"T(ρ̂-1): 均值 = {tr.mean():.2f} (偏负), 5% 分位 = {np.percentile(tr, 5):.2f}")
    print("DF 参考 5% ≈ -8.1; 若按正态(0)会错 => 须用 DF")
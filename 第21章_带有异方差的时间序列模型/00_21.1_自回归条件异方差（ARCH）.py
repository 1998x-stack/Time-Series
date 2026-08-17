# 00_21.1 自回归条件异方差（ARCH）

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def sim_arch(T, omega=0.2, alpha=0.7, rng=None):
    rng = rng or np.random.default_rng()
    h = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        h[t] = omega + alpha * y[t - 1] ** 2
        y[t] = np.sqrt(h[t]) * rng.normal()
    return y, h


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    omega, alpha = 0.2, 0.5
    T = 50000
    y, h = sim_arch(T, omega, alpha, rng)
    theo = omega / (1 - alpha)
    kurt = np.mean(((y - y.mean()) / y.std()) ** 4)
    print(f"ARCH(1): α={alpha}")
    print(f"  样本方差 = {y.var():.3f},  理论 ω/(1-α) = {theo:.3f}")
    print(f"  超额峰度 = {kurt - 3:.2f}  (厚尾 >0)")
    print(f"  条件方差 h 均值 = {h.mean():.3f}")
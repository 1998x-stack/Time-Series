# 01_21.2 扩展

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def sim_garch(T, omega, alpha, beta, rng):
    y = np.zeros(T); s2 = np.zeros(T)
    for t in range(1, T):
        s2[t] = omega + alpha * y[t - 1] ** 2 + beta * s2[t - 1]
        y[t] = np.sqrt(s2[t]) * rng.normal()
    return y, s2


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    omega, alpha, beta = 0.1, 0.1, 0.8
    T = 50000
    y, s2 = sim_garch(T, omega, alpha, beta, rng)
    theo = omega / (1 - alpha - beta)
    print(f"GARCH(1,1): α={alpha}, β={beta}, 持续性 α+β={alpha+beta:.2f}")
    print(f"  样本方差 = {y.var():.3f},  理论 = {theo:.3f}")
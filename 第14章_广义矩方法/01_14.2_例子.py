# 01_14.2 例子

import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 5000
    mu0, sig20 = 1.0, 2.0
    y = rng.normal(mu0, np.sqrt(sig20), T)

    # 矩法: E[y-mu]=0, E[(y-mu)^2 - sig2]=0
    mu_mm = y.mean()
    sig2_mm = np.mean((y - mu_mm) ** 2)
    print(f"矩法(method of moments): mu = {mu_mm:.4f}, sig2 = {sig2_mm:.4f}")
    print(f"真值: mu = {mu0}, sig2 = {sig20}")
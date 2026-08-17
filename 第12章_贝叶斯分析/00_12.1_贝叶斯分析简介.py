# 00_12.1 贝叶斯分析简介

import numpy as np
from scipy import stats


if __name__ == "__main__":
    # 高斯-高斯共轭: 已知 sigma^2, 先验 mu ~ N(mu0, tau0^2)
    rng = np.random.default_rng(2026)
    mu_true, sig2 = 3.0, 1.0
    n = 100
    y = rng.normal(mu_true, np.sqrt(sig2), n)
    mu0, tau0_2 = 0.0, 4.0

    ybar = y.mean()
    kappa_data = n / sig2
    kappa_prior = 1.0 / tau0_2
    mu_post = (kappa_data * ybar + kappa_prior * mu0) / (kappa_data + kappa_prior)
    kappa_post = kappa_data + kappa_prior

    print(f"数据精度 {kappa_data:.2f}  (样本均值 {ybar:.3f})")
    print(f"先验精度 {kappa_prior:.2f}  (先验均值 {mu0})")
    print(f"后验均值 = {mu_post:.4f} (精度加权)")
    print(f"后验精度 = {kappa_post:.2f} = 先验+数据")

    # 数据增大 -> 后验趋近样本均值
    for nn in (10, 100, 1000):
        yb = y[:nn].mean()
        mp = (nn / sig2 * yb + kappa_prior * mu0) / (nn / sig2 + kappa_prior)
        print(f"  n={nn:4d}: 后验均值={mp:.3f} → 样本均值 {yb:.3f}")
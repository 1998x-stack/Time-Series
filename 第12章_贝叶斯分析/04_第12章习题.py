# 04_第12章习题

import numpy as np


if __name__ == "__main__":
    # 习题1: 共轭后验
    n, ybar, sig2, mu0, tau20 = 50, 2.2, 1.0, 0.0, 4.0
    kappa = n / sig2 + 1 / tau20
    mu_post = (n / sig2 * ybar + 1 / tau20 * mu0) / kappa
    print(f"习题1: 后验均值 = {mu_post:.4f}, 后验方差 = {1/kappa:.5f}")

    # 习题3: MH 不需要归一化 (接受率用比值)
    print("习题3: MH 接受概率用比值(π(θ*)/π(θ)), 消去归一化常数")

    # 小型 MH 演示
    rng = np.random.default_rng(1)
    target_mu, target_sd = 2.0, 0.5
    def logt(x):
        return -0.5 * ((x - target_mu) / target_sd) ** 2
    theta = 0.0; draws = []
    for i in range(5000):
        ts = theta + rng.normal(0, 0.5)
        if rng.uniform() < min(1.0, np.exp(logt(ts) - logt(theta))):
            theta = ts
        draws.append(theta)
    print("习题3(小MH): 均值 =", round(np.mean(draws), 3), " 应≈2")
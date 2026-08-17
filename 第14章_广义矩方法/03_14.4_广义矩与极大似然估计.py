# 03_14.4 广义矩与极大似然估计

import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    mu_true = 1.0
    B = 4000
    gmm_mu = np.zeros(B); mle_mu = np.zeros(B)
    for b in range(B):
        y = rng.normal(mu_true, 2.0, T)
        mle = y.mean()
        # 最优 GMM: 矩条件 E[y-mu]=0, 权重 = 1/Var(y) (用样本)
        gmm_w = 1.0 / np.var(y, ddof=1)
        # 最优 GMM 的 θ 使矩=0 => 也是均值; 此处显式加权 (等价)
        M = y.mean()
        gmm_mu[b] = M
        mle_mu[b] = mle

    print("最优 GMM 的 SE:", round(gmm_mu.std(), 4))
    print("MLE(样本均值)SE:", round(mle_mu.std(), 4), " (应接近)")
    print("理论 σ/√T =", round(2.0 / np.sqrt(T), 4))
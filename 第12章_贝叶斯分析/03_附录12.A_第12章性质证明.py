# 03_附录12.A 第12章性质证明

import numpy as np
from scipy import stats


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    y = rng.normal(2.0, 1.0, 100)
    mu0, tau20 = 0.0, 10.0

    # 数值后验核: prior(mu) * lik(mu)
    grid = np.linspace(-4, 8, 60001)
    kern = np.exp([stats.norm.logpdf(m, 0, np.sqrt(tau20)) +
                   np.sum(stats.norm.logpdf(y, m, 1.0)) for m in grid])
    Z = np.trapezoid(kern, grid)              # 归一化常数
    post = kern / Z
    # 数值后验矩
    mean_num = np.trapezoid(grid * post, grid)
    var_num = np.trapezoid((grid - mean_num) ** 2 * post, grid)

    # 解析共轭后验
    mu_post = (len(y) / 1.0 * y.mean() + 1 / tau20 * mu0) / (len(y) / 1.0 + 1 / tau20)
    var_post = 1 / (len(y) / 1.0 + 1 / tau20)

    print("归一化后 ∫p(μ|y)dμ =", round(np.trapezoid(post, grid), 6), " (应=1)")
    print("数值后验 均值 =", round(mean_num, 4), " 解析 =", round(mu_post, 4))
    print("数值后验 方差 =", round(var_num, 4), " 解析 =", round(var_post, 4))
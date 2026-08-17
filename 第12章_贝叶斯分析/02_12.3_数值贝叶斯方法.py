# 02_12.3 数值贝叶斯方法

import numpy as np
from scipy import stats


def mh_normal_posterior(log_target, start, proposal_sd, n, rng):
    """随机游走 Metropolis-Hastings 采样 (1 维)。"""
    draws = np.zeros(n)
    theta = start
    accept = 0
    for i in range(n):
        theta_star = theta + rng.normal(0, proposal_sd)
        alpha = np.exp(log_target(theta_star) - log_target(theta))
        if rng.uniform() < min(1.0, alpha):
            theta = theta_star
            accept += 1
        draws[i] = theta
    return draws, accept / n


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    mu_true = 2.0
    y = rng.normal(mu_true, 1.0, 100)

    # 先验 mu ~ N(0, 10)
    mu0, tau20 = 0.0, 10.0
    mu_post = (100 / 1.0 * y.mean() + 1 / 10.0 * mu0) / (100 / 1.0 + 1 / 10.0)
    var_post = 1 / (100 / 1.0 + 1 / 10.0)
    print("解析后验: mu =", round(mu_post, 4), " sd =", round(np.sqrt(var_post), 4))

    def logp(mu):
        return stats.norm.logpdf(mu, 0, np.sqrt(tau20)) + np.sum(
            stats.norm.logpdf(y, mu, 1.0))

    draws, acc = mh_normal_posterior(logp, 1.0, 0.2, 20000, rng)
    draws = draws[2000:]          # 预热
    print("MH 后验:  mean =", round(draws.mean(), 4), " sd =", round(draws.std(), 4))
    print("接受率 =", round(acc, 3))
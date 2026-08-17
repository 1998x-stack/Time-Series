# 02_22.3 独立同分布的混合分布

import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def em_mixture(X, k=2, iters=300, seed=1):
    rng = np.random.default_rng(seed)
    n = len(X)
    pi = np.full(k, 1 / k)
    mu = np.quantile(X, np.linspace(0.2, 0.8, k))
    sd = np.full(k, X.std())
    for _ in range(iters):
        w = pi[:, None] * norm.pdf(X, mu[:, None], sd[:, None])
        g = w / w.sum(0)                 # 后验 (k,n)
        nk = g.sum(1)
        mu = (g @ X) / nk
        sd = np.sqrt(np.sum(g * (X - mu[:, None]) ** 2, 1) / nk)
        pi = nk / n
    order = np.argsort(mu)
    return pi[order], mu[order], sd[order]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 20000
    w = 0.6
    mu_t = np.array([-2.0, 2.0]); sd_t = np.array([0.8, 1.2])
    c = (rng.random(n) < w).astype(int)
    mu_c = mu_t[c]; sd_c = sd_t[c]
    X = mu_c + sd_c * rng.normal(size=n)

    pi, mu, sd = em_mixture(X)
    print("EM 估计(按均值排序): 权重=", np.round(pi, 3), " 真=", np.array([w, 1 - w]))
    print("      均值 =", np.round(mu, 3), "  真 =", mu_t)
    print("      标准差 =", np.round(sd, 3), " 真 =", sd_t)
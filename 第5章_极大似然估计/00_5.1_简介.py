# 00_5.1 简介

"""
Lecture: /第5章 极大似然估计
Content: 00_5.1 简介
"""

import numpy as np


def loglik_gauss(x, mu, sig2):
    n = len(x)
    return -0.5 * n * np.log(2 * np.pi * sig2) - np.sum((x - mu) ** 2) / (2 * sig2)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 500
    mu0, sig02 = 1.0, 2.0
    x = rng.normal(mu0, np.sqrt(sig02), n)

    # MLE
    mle_mu = x.mean()
    mle_sig2 = np.mean((x - mle_mu) ** 2)
    print("MLE 均值 =", round(mle_mu, 4), " (真 1.0)")
    print("MLE 方差 =", round(mle_sig2, 4), " (真 2.0, 偏差 =", round(mle_sig2 - 2.0, 4), ")")

    # 得分数在 MLE 处≈0: s_mu = n(mu-xbar)/sig2
    print("得分(μ) 在 MLE 处:", round(n * (mle_mu - mle_mu) / mle_sig2, 6), " (0)")

    # 信息矩阵约 1/Var: 观察信息 -d2ℓ/dμ2 = n/sig2
    info_mu = n / mle_sig2
    print("信息 I_mu ≈ n/sig2 =", round(info_mu, 2), " => SE ≈", round(1 / np.sqrt(info_mu), 4))

    # 比较 MLE 与真实参数处的对数似然 (MLE 应更高)
    print("ℓ(MLE) =", round(loglik_gauss(x, mle_mu, mle_sig2), 2),
          " > ℓ(真) =", round(loglik_gauss(x, mu0, sig02), 2), "?",
          loglik_gauss(x, mle_mu, mle_sig2) > loglik_gauss(x, mu0, sig02))
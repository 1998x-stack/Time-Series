# 07_5.8 极大似然估计的统计推断

"""
Lecture: /第5章 极大似然估计
Content: 07_5.8 极大似然估计的统计推断
"""

import numpy as np


def simulate_ar1(phi, sigma, n, rng):
    burn = 300
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(1, total):
        x[t] = phi * x[t - 1] + eps[t]
    return x[burn:]


def neg_loglik_ar1(par, x):
    phi, sig2 = par
    if sig2 <= 0:
        return 1e30
    e = x[1:] - phi * x[:-1]
    return (len(x) - 1) / 2 * np.log(2 * np.pi * sig2) + np.sum(e ** 2) / (2 * sig2)


def numerical_hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp, xpm = x.copy(), x.copy()
            xmp, xmm = x.copy(), x.copy()
            xpp[i] += h; xpp[j] += h
            xpm[i] += h; xpm[j] -= h
            xmp[i] -= h; xmp[j] += h
            xmm[i] -= h; xmm[j] -= h
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
    return H


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi0, sig0 = 0.6, 1.0
    n = 4000
    x = simulate_ar1(phi0, sig0, n, rng)

    phi_hat = np.dot(x[1:], x[:-1]) / np.dot(x[:-1], x[:-1])
    res = np.array([phi_hat, phi_hat])
    sig_hat = np.mean((x[1:] - phi_hat * x[:-1]) ** 2)

    # 负 Hessian = 信息矩阵; SE = sqrt(对角逆)
    H = numerical_hessian(lambda p: neg_loglik_ar1(p, x), np.array([phi_hat, sig_hat]))
    cov = np.linalg.inv(H)
    se_phi = np.sqrt(cov[0, 0])
    se_analytic = np.sqrt((1 - phi_hat ** 2) / n)
    ci = (phi_hat - 1.96 * se_phi, phi_hat + 1.96 * se_phi)
    print(f"phi_hat = {phi_hat:.4f},  SE(观测信息) = {se_phi:.4f},  SE(解析) = {se_analytic:.4f}")
    print(f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}],  含真值 0.6? {ci[0] <= 0.6 <= ci[1]}")
    print("t =", round(phi_hat / se_phi, 3), "(远大于 2, H0:phi=0 被拒)")
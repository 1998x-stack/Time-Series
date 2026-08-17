# 01_7.2 序列相关观测的极限定理

"""
Lecture: /第7章 渐近分布理论
Content: 01_7.2 序列相关观测的极限定理
"""

import numpy as np


def simulate_ar1(phi, sigma, T, rng):
    burn = 300
    total = T + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    for t in range(1, total):
        x[t] = phi * x[t - 1] + eps[t]
    return x[burn:]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi, sigma = 0.7, 1.0
    T = 400
    B = 4000

    lam_theory = sigma ** 2 / (1 - phi) ** 2

    xbar = np.array([simulate_ar1(phi, sigma, T, rng).mean() for _ in range(B)])
    lam_emp = T * xbar.var()
    print("经验 T*Var(xbar) =", round(lam_emp, 3),
          "  理论 λ = σ²/(1-φ)² =", round(lam_theory, 3))
    print("iid 情形 σ² =", sigma ** 2, " => λ/σ² 经验 =", round(lam_emp / sigma ** 2, 2),
          " 理论 1/(1-φ)² =", round(1 / (1 - phi) ** 2, 2))
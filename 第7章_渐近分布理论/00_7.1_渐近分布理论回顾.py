# 00_7.1 渐近分布理论回顾

"""
Lecture: /第7章 渐近分布理论
Content: 00_7.1 渐近分布理论回顾
"""

import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    mu, sig = 3.0, 2.0
    T = 200
    B = 20000

    # CLT: sqrt(T)(xbar - mu)/sig ~ N(0,1)
    z = np.array([np.sqrt(T) * (rng.normal(mu, sig, T).mean() - mu) / sig
                  for _ in range(B)])
    print("CLT: 经验 E[z] =", round(z.mean(), 3), "(应0),  Var[z] =", round(z.var(), 3), "(应1)")
    q = np.percentile(z, [2.5, 50, 97.5])
    print("CLT: 经验 2.5/50/97.5 分位 =", np.round(q, 3), " vs N(0,1) [-1.96,0,1.96]")

    # LLN: 样本均值随 T 收敛
    print("LLN: |xbar - mu| 随 T 递减:")
    for TT in (20, 200, 2000):
        print(f"  T={TT:5d} |xbar-mu|={abs(rng.normal(mu,sig,TT).mean()-mu):.4f}")

    # Delta: sqrt(T)(log(xbar) - log(mu)) -> N(0, sig^2/mu^2)
    Td = 200
    lz = np.array([np.sqrt(Td) * (np.log(rng.normal(mu, sig, Td).mean()) - np.log(mu))
                   for _ in range(B)])
    print("Delta: Var(√T log x̄) =", round(lz.var(), 3),
          " vs σ²/μ² =", round(sig ** 2 / mu ** 2, 3))
# 09_附录5.A_第5章性质证明

"""
Lecture: /第5章 极大似然估计
Content: 09_附录5.A 第5章性质证明
"""

import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 50
    mu0, sig02 = 0.0, 1.0
    B = 20000

    # 在真值处计算得分与观测信息
    scores = np.zeros(B)
    info = np.zeros(B)
    for b in range(B):
        x = rng.normal(mu0, np.sqrt(sig02), n)
        s_b = n * (x.mean() - mu0) / sig02          # dℓ/dμ
        scores[b] = s_b
        info[b] = n / sig02                          # -ℓ''(μ) 恒为常数

    print("E[得分] =", round(scores.mean(), 5), " (应≈0)")
    print("Var(得分) =", round(scores.var(), 4), " 信息 n/σ² =", n / sig02)
    print("验证: Var(s) ≈ I =", round(scores.var(), 4) / (n / sig02),
          " (应接近1)")
    # 得分分布近似 N(0, I)
    print("得分标准差", round(scores.std(), 3), " vs sqrt(I)=", round(np.sqrt(n / sig02), 3))
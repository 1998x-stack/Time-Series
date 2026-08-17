# 03_第7章习题

"""
Lecture: /第7章 渐近分布理论
Content: 03_第7章习题
"""

import numpy as np


if __name__ == "__main__":
    # 习题1: AR(1) 长程方差与 Var(xbar)
    phi, sig2, T = 0.5, 1.0, 100
    lam = sig2 / (1 - phi) ** 2
    var_xbar = lam / T
    print(f"习题1: λ = {lam}, Var(bar x) = λ/T = {var_xbar}")

    # 习题2: iid CLT 区间
    z = 1.96 / np.sqrt(T)
    print(f"习题2: iid N(0,1), 95% 区间 x_bar ± {z:.4f}")

    # 习题3: Delta exp
    vth = 0.01
    theta = 0.5
    vexp = (np.exp(theta) ** 2) * vth
    print(f"习题3: Var(e^theta) ≈ e^{{2θ}}·0.01 = {vexp:.4f}")
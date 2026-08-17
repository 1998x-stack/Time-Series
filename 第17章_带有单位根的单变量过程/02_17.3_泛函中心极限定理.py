# 02_17.3 泛函中心极限定理

import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    B = 20000
    T = 20000
    # X_T(1) = (1/sqrt T) sum_{i<=T} e_i
    X = np.array([np.sum(rng.normal(0, 1, T)) / np.sqrt(T) for _ in range(B)])
    print(f"X_T(1): 均值 {X.mean():.3f} (应0), 方差 {X.var():.3f} (应1)")
    q = np.percentile(X, [2.5, 97.5])
    print("  95% 区间", np.round(q, 3), " vs ±1.96 (N(0,1))")
    # 布朗方差: 在 s=0.5, X_T(0.5) 方差≈0.5
    X5 = np.array([np.sum(rng.normal(0, 1, T // 2)) / np.sqrt(T) for _ in range(4000)])
    print(f"X_T(0.5): 方差 {X5.var():.3f} (应≈0.5, 布朗的特征)")
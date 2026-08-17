# 05_附录15.A 第15章部分公式的推导

import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    sig2 = 1.0
    B = 20000
    tvals = (10, 50, 100)
    for t in tvals:
        # Var(y_t) = t sig2 (理论); 用 MC 验证
        yt = np.array([np.sum(rng.normal(0, 1, t)) for _ in range(B)])
        ratio = yt.var() / (t * sig2)
        print(f"t={t:3d}: 经验 Var(y_t)={yt.var():.2f}  理论 tσ²={t}  比值={ratio:.3f}")
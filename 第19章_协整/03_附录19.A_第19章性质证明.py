# 03_附录19.A 第19章性质证明

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def beta_hat(T, rng, beta=1.5):
    y2 = np.cumsum(rng.normal(0, 1, T))
    y1 = beta * y2 + rng.normal(0, 0.5, T)
    b = np.dot(y2, y1) / np.dot(y2, y2)
    return b


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    B = 1500
    for T in (200, 800, 3200):
        bs = np.array([beta_hat(T, rng) for _ in range(B)])
        sd = bs.std()
        print(f"T={T:5d}: SD(β̂)={sd:.4f}   T·SD={T*sd:.3f}  (超一致的常量化)")
    print("协整系数估计 SD~1/T (超一致); 一般回归则 1/√T")
# 00_16.1 简单时间趋势模型 OLS 估计的渐近分布

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def ols_trend(T, rng, mu=0.0, delta=0.5):
    t = np.arange(T).astype(float)
    y = mu + delta * t + rng.normal(0, 1, T)
    X = np.column_stack([np.ones(T), t])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b   # [mu_hat, delta_hat]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    B = 4000
    for T in (200, 800, 3200):
        ds = np.array([ols_trend(T, rng)[1] for _ in range(B)])
        sd = ds.std()
        # 理论 ~ sqrt(12 sigma2 / T^3)
        theo = np.sqrt(12.0 / T ** 3)
        print(f"T={T:5d}: SD(δ̂)={sd:.5f}  理论=√(12σ²/T³)={theo:.5f}  比值={sd/theo:.2f}")
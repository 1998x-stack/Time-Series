# 02_16.3 含确定性时间趋势的自回归过程的渐近推断

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def fit_ar_trend(T, rng, gamma=0.6, delta=0.02, mu=0.5):
    t = np.arange(T).astype(float)
    y = np.zeros(T)
    for tt in range(1, T):
        y[tt] = mu + delta * tt + gamma * y[tt - 1] + rng.normal(0, 1)
    X = np.column_stack([np.ones(T - 1), t[1:], y[:-1]])
    b = np.linalg.lstsq(X, y[1:], rcond=None)[0]
    return b  # [mu, delta, gamma]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    B = 2000
    for T in (300, 1200):
        Bg = np.array([fit_ar_trend(T, rng) for _ in range(B)])
        print(f"T={T}: γ̂ SE={Bg[:,2].std():.4f}, δ̂ SE={Bg[:,1].std():.5f}, "
              f"均值 γ̂={Bg[:,2].mean():.3f} (真0.6)")
# 05_第6章习题

"""
Lecture: /第6章 谱分析
Content: 05_第6章习题
"""

import numpy as np


if __name__ == "__main__":
    # 习题1: AR(1) f(0)
    phi, sig2 = 0.5, 1.0
    f0 = sig2 / (2 * np.pi * (1 - phi) ** 2)
    print(f"习题1: f(0) = {f0:.4f} (理论 1/(2pi*0.25)=0.6366)")

    # 习题2: 周期换算
    lam = 0.6
    print(f"习题2: 周期 = 2pi/{lam} = {2*np.pi/lam:.2f} 季 ≈ {2*np.pi/lam/4:.2f} 年")

    # 习题3: 演示周期图不一致 -> 平滑修正
    rng = np.random.default_rng(2026)
    T = 2000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.7 * x[t - 1] + rng.normal(size=1)[0]
    # 周期图在相邻频率的波动 (方差不消失)
    I = np.abs(np.fft.fft(x - x.mean())) ** 2 / T
    neigh = I[200:206]
    print("习题3: 相邻频率周期图像元波动 (不一致):", np.round(neigh, 2),
          " (单点方差大, 需平滑)")
# 01_15.2 为什么考虑线性时间趋势和单位根?

import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 60
    shock = 1.0

    # 同一噪声: 一条有单位根冲击, 一条无(对照)
    noise = rng.normal(0, 0.2, T + 1)
    rw_shock = np.zeros(T + 1); rw_ctl = np.zeros(T + 1)
    for t in range(1, T + 1):
        s = shock if t == 30 else 0.0
        rw_shock[t] = rw_shock[t - 1] + noise[t] + s
        rw_ctl[t] = rw_ctl[t - 1] + noise[t]
    diff = rw_shock - rw_ctl
    print("单位根: 冲击 vs 无冲击之差 (应≈1, 持久):")
    for h in (1, 5, 20, 30):
        print(f"  冲击后 {h:2d} 期: diff ≈ {diff[30+h]:+.2f}")
    print("TS(白噪声): 冲击只影响当期, 之后回到 0")
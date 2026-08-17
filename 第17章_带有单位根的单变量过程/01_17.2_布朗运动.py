# 01_17.2 布朗运动

import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    N = 1000          # 网格段
    B = 5000
    s_grid = np.linspace(0, 1, N + 1)
    # 多复制布朗: 用累积增量
    incr = rng.normal(0, np.sqrt(1.0 / N), (B, N))
    W = np.cumsum(incr, axis=1)
    W = np.column_stack([np.zeros(B), W])

    for s in (0.1, 0.5, 0.9, 1.0):
        idx = int(np.argmin(np.abs(s_grid - s)))
        emp = W[:, idx].var()
        print(f"s={s:.1f}: 经验 Var(W(s))={emp:.3f}  理论 s={s}  比值={emp/s:.3f}")
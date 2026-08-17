# 06_附录9.A 第9章性质证明

"""
Lecture: /第9章 线性联立方程系统
Content: 06_附录9.A 第9章性质证明
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def dgp(T, rng):
    x1 = rng.normal(size=T); x2 = rng.normal(size=T)
    e2 = rng.normal(size=T)
    e1 = 0.6 * e2 + 0.8 * rng.normal(size=T)
    y2 = 1.0 * x1 + 0.7 * x2 + e2
    y1 = 0.8 * y2 + 0.5 * x1 + e1
    return y1, y2, x1, x2, e1


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 20000
    y1, y2, x1, x2, e1 = dgp(T, rng)
    print("正交条件 corr(x2, e1) =", round(np.corrcoef(x2, e1)[0, 1], 4), "(应≈0)")
    print("相关条件 corr(x2, y2) =", round(np.corrcoef(x2, y2)[0, 1], 4), "(应≠0)")

    # 2SLS 随 T 一致
    for TT in (100, 1000, 20000):
        y1t, y2t, x1t, x2t = (v[:TT] for v in (y1, y2, x1, x2))
        Z = np.column_stack([x1t, x2t])
        y2h = Z @ np.linalg.lstsq(Z, y2t, rcond=None)[0]
        a = np.linalg.lstsq(np.column_stack([y2h, x1t]), y1t, rcond=None)[0][0]
        print(f"  T={TT:6d}: 2SLS alpha={a:.4f} (真0.8)")
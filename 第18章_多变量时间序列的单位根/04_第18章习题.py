# 04_第18章习题

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def dw(e):
    return np.sum(np.diff(e) ** 2) / np.sum(e ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 400
    y = np.cumsum(rng.normal(0, 1, T)); x = np.cumsum(rng.normal(0, 1, T))
    # 水平回归
    X = np.column_stack([np.ones(T), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    R2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"水平回归: R²={R2:.2f}, DW={dw(e):.2f} (伪回归特征)")

    # 差分回归
    dy, dx = np.diff(y), np.diff(x)
    Xd = np.column_stack([np.ones(T - 1), dx])
    bd = np.linalg.lstsq(Xd, dy, rcond=None)[0]
    ed = dy - Xd @ bd
    print(f"差分回归: DW={dw(ed):.2f} (应≈2, 真关系不存在)")
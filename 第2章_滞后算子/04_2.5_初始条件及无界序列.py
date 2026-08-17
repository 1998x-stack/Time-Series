# 04_2.5 初始条件及无界序列

"""
Lecture: /第2章 滞后算子
Content: 04_2.5 初始条件及无界序列
"""

import numpy as np


def backward_unbounded(phi: float, w: np.ndarray) -> np.ndarray:
    """向后(因果)解 y_t = phi y_{t-1} + w_t, 从 y_0 = 0。φ>1 时无界。"""
    n = len(w)
    y = np.empty(n)
    prev = 0.0
    for t in range(n):
        prev = phi * prev + w[t]
        y[t] = prev
    return y


def forward_bounded(phi: float, w: np.ndarray) -> np.ndarray:
    """向前(有界)解: 由远端零边界回代 y_{t-1} = (y_t - w_t)/φ。φ>1 时有界。"""
    n = len(w)
    y = np.zeros(n)
    for t in range(n - 1, 0, -1):
        y[t - 1] = (y[t] - w[t]) / phi
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 100
    phi = 1.2
    w = rng.normal(size=n)

    yc = backward_unbounded(phi, w)
    yb = forward_bounded(phi, w)

    # 因果解满足 (1 - phi L)y = w 精确
    resid_c = yc[1:] - phi * yc[:-1] - w[1:]
    print("因果解: (1-phiL)y 与 w 最大差 =", np.max(np.abs(resid_c)))

    # 有界解应保持有界, 因果解指数增长
    print(f"最大 |因果解| = {np.max(np.abs(yc)):.4f}  最大 |有界解| = {np.max(np.abs(yb)):.4f}")
    print("增长对比 (t 与对应 |解|):")
    for t in (5, 20, 50, 90):
        print(f"  t={t:3d}  |因果|={abs(yc[t]):.3e}   |有界|={abs(yb[t]):.4f}")

    # 两条解之差是齐次解 C*phi^t (验证比值近似常数)
    d = yb - yc
    factor = d[-1] / phi ** (n - 1)
    residual_scale = np.max(np.abs(d - factor * phi ** np.arange(n)))
    print("(yb-yc) 与 C*phi^t 拟合残差:", residual_scale)
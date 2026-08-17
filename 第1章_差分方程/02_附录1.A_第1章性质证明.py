# 02_附录1.A 第1章性质证明

"""
Lecture: /第1章 差分方程
Content: 02_附录1.A 第1章性质证明
"""

import numpy as np


def _recursive_solution(phi: float, x0: float, w: np.ndarray) -> np.ndarray:
    """向前递推 y_t = phi*y_{t-1} + w_t。"""
    T = len(w)
    x = np.empty(T)
    prev = x0
    for t in range(T):
        prev = phi * prev + w[t]
        x[t] = prev
    return x


def _closed_solution(phi: float, x0: float, w: np.ndarray) -> np.ndarray:
    """显式解 y_t = phi^{t+1}x0 + sum_{j=0..t} phi^{t-j} w_j (0 起)。"""
    T = len(w)
    impact = phi ** np.arange(1, T + 1) * x0
    psi = phi ** np.arange(T)
    past = np.convolve(w, psi)[:T]
    return impact + past


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    x0 = 1.0
    for phi in (0.6, 0.9, 1.0, 1.05):
        w = rng.normal(size=T)
        direct = _recursive_solution(phi, x0, w)
        closed = _closed_solution(phi, x0, w)
        print(f"[phi={phi}] 递推 vs 闭式 max-diff = {np.max(np.abs(direct - closed)):.3e} | "
              f"初值分量 phi^T = {phi**T:.6e}")
    # 性质数列: |phi|<1 => 初值分量极小(收敛); |phi|=1 不衰减; |phi|>1 发散
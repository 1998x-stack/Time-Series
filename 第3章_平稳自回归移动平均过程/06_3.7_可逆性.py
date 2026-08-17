# 06_3.7 可逆性

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 06_3.7 可逆性
"""

import numpy as np


def lag(x: np.ndarray, k: int = 1) -> np.ndarray:
    """滞后算子: 返回左移 k 期、右侧补零的序列。"""
    n = len(x)
    if k >= n:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[k:] = x[: n - k]
    return out


def convert_trunc(theta: float, J: int, x: np.ndarray) -> np.ndarray:
    """用截断 AR(∞) 还原残差: eps_hat = sum_{j=0..J} (-theta)^j L^j x。"""
    total = np.zeros_like(x)
    for j in range(J + 1):
        total += (-theta) ** j * lag(x, j)
    return total


def simulate_ma(theta: float, sigma: float, n: int, rng) -> tuple:
    """MA(1): x_t = eps_t + theta eps_{t-1}。返回 (x, eps)。"""
    burn = 200
    total = n + burn
    eps = rng.normal(0.0, sigma, total)
    x = np.zeros(total)
    prev_eps = 0.0
    for t in range(total):
        x[t] = eps[t] + theta * prev_eps
        prev_eps = eps[t]
    return x[burn:], eps[burn:]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    sigma = 1.0
    n = 4000

    # 可逆 MA(1): |theta|<1, 还原误差随 J 减小
    theta = 0.5
    x, eps = simulate_ma(theta, sigma, n, rng)
    print(f"可逆 MA(1) theta={theta} (c_j=(-theta)^j), 还原误差随 J 递减:")
    for J in (2, 8, 32):
        rec = convert_trunc(theta, J, x)
        e = np.max(np.abs(rec[30: n] - eps[30: n]))
        print(f"  J={J:2d}  误差 = {e:.3e}")

    # 不可逆 MA(1): |theta|>1, 因果还原发散
    theta2 = 2.0
    x2, eps2 = simulate_ma(theta2, sigma, n, rng)
    print(f"不可逆 MA(1) theta={theta2} (|theta|>1, 还原发散):")
    for J in (2, 8, 32):
        rec = convert_trunc(theta2, J, x2)
        e = np.max(np.abs(rec[30: n] - eps2[30: n]))
        print(f"  J={J:2d}  误差 = {e:.3e}  (误差随 J 增大)")
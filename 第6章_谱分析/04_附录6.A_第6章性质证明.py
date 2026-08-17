# 04_附录6.A 第6章性质证明

"""
Lecture: /第6章 谱分析
Content: 04_附录6.A 第6章性质证明
"""

import numpy as np


def ar1_spectrum(phi, sigma, lam):
    return (sigma ** 2 / (2 * np.pi)) / np.abs(1 - phi * np.exp(-1j * lam)) ** 2


if __name__ == "__main__":
    phi, sigma = 0.7, 1.0
    grid = np.linspace(-np.pi, np.pi, 8001)
    f = ar1_spectrum(phi, sigma, grid)
    dlam = grid[1] - grid[0]

    # 非负性 + 偶性(周期性)
    print("min f =", round(f.min(), 6), " (应>=0), f(λ)=f(-λ) 误差:",
          round(np.max(np.abs(f - f[::-1])), 12))

    # 逆变换恢复 gamma(h)
    print("逆变换恢复 gamma:")
    for h in range(5):
        gamma_est = np.trapezoid(f * np.exp(1j * h * grid), grid)
        gamma_true = sigma ** 2 * phi ** h / (1 - phi ** 2)
        print(f"  h={h}: 重收={np.real(gamma_est):.4f}  闭式={gamma_true:.4f}")

    # 面积 = 方差
    area = np.trapezoid(f, grid)
    print("∫f dλ =", round(area, 4), " = γ(0) =", round(sigma ** 2 / (1 - phi ** 2), 4))
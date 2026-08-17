# 10_第5章习题

"""
Lecture: /第5章 极大似然估计
Content: 10_第5章习题
"""

import numpy as np


if __name__ == "__main__":
    # 习题1: 给定过统计量
    num, den, n = 120.0, 150.0, 200
    phi_hat = num / den
    se = np.sqrt((1 - phi_hat ** 2) / n)
    print(f"习题1: phi_hat = {phi_hat:.3f}, SE = {se:.4f}")

    # 习题2: 可逆性约束说明 (数值: 用界)
    print("习题2: 约束 |theta|<1 保证可逆(根在单位圆外), 过滤掉等价折叠表示")

    # 习题3: 用模拟验证得分方差=信息
    rng = np.random.default_rng(2026)
    nn, B = 50, 20000
    s = np.array([nn * (rng.normal(size=nn).mean()) for _ in range(B)])  # 真值 μ0=0
    info = N / 1.0 if False else nn / 1.0
    print(f"习题3: Var(得分)={s.var():.3f} vs 信息 n/σ²={info:.2f} (≈相等)")
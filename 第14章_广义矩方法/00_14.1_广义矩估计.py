# 00_14.1 广义矩估计

import numpy as np


def gmm(theta_guess, y, x, z, W):
    """目标: J = mbar' W mbar, m_i = z_i(y_i - x_i θ)。θ 标量。"""
    # 样本矩 mbar(θ) = (1/T) Σ z_i (y_i - x_i θ)
    Mzx = (z * x[:, None]).mean(axis=0)   # 每工具 L 维度: E[z_i x_i]
    Mzy = (z * y[:, None]).mean(axis=0)   # E[z_i y_i]
    mbar = Mzy - Mzx * theta_guess
    return np.asarray(mbar) @ W @ np.asarray(mbar)


def gmm_estimate(y, x, z, W):
    Mzx = (z * x[:, None]).mean(axis=0)
    Mzy = (z * y[:, None]).mean(axis=0)
    # min θ: mbar'W mbar -> θ = (Mzx' W Mzx)^{-1} Mzx' W Mzy
    a = float(Mzx @ W @ Mzx)
    b = float(Mzx @ W @ Mzy)
    return b / a


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 3000
    x = rng.normal(0, 1, T)
    e = rng.normal(0, 1, T)
    y = x * 2.0 + e
    # 两个工具 (z1=x, z2=x^2)
    z = np.column_stack([x, x ** 2])

    # 第一步: 等权 W=I
    W1 = np.eye(2)
    th1 = gmm_estimate(y, x, z, W1)
    # 第二步: 由 th1 计算最优权重
    u = y - x * th1
    Om = (z * u[:, None]).T @ (z * u[:, None]) / T
    W2 = np.linalg.inv(Om + 1e-8 * np.eye(2))
    th2 = gmm_estimate(y, x, z, W2)
    print("GMM 等权 θ =", round(th1, 4), "  最优两阶段 θ =", round(th2, 4),
          "  真值 = 2.0")
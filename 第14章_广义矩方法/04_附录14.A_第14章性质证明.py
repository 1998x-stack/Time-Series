# 04_附录14.A 第14章性质证明

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def gmm_est(y, x, z, W):
    Mzx = (z * x[:, None]).mean(axis=0)
    Mzy = (z * y[:, None]).mean(axis=0)
    a = float(Mzx @ W @ Mzx); b = float(Mzx @ W @ Mzy)
    return b / a


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 800
    B = 1500
    L = 3
    th_eq = np.zeros(B); th_opt = np.zeros(B)
    for b in range(B):
        x = rng.normal(0, 1, T); e = rng.normal(0, 1, T)
        y = x * 2.0 + e
        z = np.column_stack([x, x ** 2, x ** 3])
        # 等权
        te = gmm_est(y, x, z, np.eye(L))
        # 最优两阶段
        u = y - x * te
        Om = (z * u[:, None]).T @ (z * u[:, None]) / T
        to = gmm_est(y, x, z, np.linalg.inv(Om + 1e-8 * np.eye(L)))
        th_eq[b] = te; th_opt[b] = to

    print("等权 GMM SE:", round(th_eq.std(), 4))
    print("最优 GMM SE:", round(th_opt.std(), 4), " (应 ≤ 等权)")
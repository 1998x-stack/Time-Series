# 05_第14章习题

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 3000
    x = rng.normal(0, 1, T); e = rng.normal(0, 1, T)
    y = x * 2.0 + e
    z = np.column_stack([x, x ** 2, x ** 3]); L = z.shape[1]

    # 等权
    Mzx = (z * x[:, None]).mean(0); Mzy = (z * y[:, None]).mean(0)
    th1 = float(Mzx @ Mzy) / float(Mzx @ Mzx)
    # 最优两步
    u = y - x * th1
    Om = (z * u[:, None]).T @ (z * u[:, None]) / T
    W = np.linalg.inv(Om + 1e-8 * np.eye(L))
    th2 = (float(Mzx @ W @ Mzy) / float(Mzx @ W @ Mzx))
    # J
    mbar = Mzy - Mzx * th2
    J = T * float(mbar @ W @ mbar)
    p = 1 - stats.chi2.cdf(J, L - 1)
    print(f"习题1/3: 最优 GMM θ={th2:.3f}, 与 MLE 效率同(正确设定)")
    print(f"习题2: J = {J:.2f}, df={L-1}, p = {p:.3f} (真模型通过)")
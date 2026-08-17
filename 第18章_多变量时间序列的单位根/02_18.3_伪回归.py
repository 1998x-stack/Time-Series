# 02_18.3 伪回归

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def dw(resid):
    e = np.asarray(resid)
    return np.sum((np.diff(e)) ** 2) / np.sum(e ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 500
    y = np.cumsum(rng.normal(0, 1, T))
    x = np.cumsum(rng.normal(0, 1, T))     # 与 y 独立

    X = np.column_stack([np.ones(T), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    k = 2
    s2 = np.sum(e ** 2) / (T - k)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    t = b[1] / se
    R2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    d = dw(e)
    print(f"伪回归: β̂={b[1]:.3f}, t={t:.1f} (极大), R²={R2:.2f} (高), "
          f"DW={d:.2f} (≈0)")
    print("=> 显著但虚假; 真关系不存在(差分/协整才是正道)")
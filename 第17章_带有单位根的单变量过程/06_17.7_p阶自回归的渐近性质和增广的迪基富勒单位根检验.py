# 06_17.7 p阶自回归与自然增广检验(ADF)

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def adf_stat(y, p=2):
    """ADF: Δy_t = a + g y_{t-1} + Σ_{j=1..p} b_j Δy_{t-j} + u. 返回 g 的 t。"""
    T = len(y)
    dy = np.diff(y)                      # dy[k] = y[k+1]-y[k]
    # 实际时刻 t: t-1 = dy 的 index。回归在 t = p+1..T (0-based dy index p..T-2 对应 Δy)
    lo, hi = p, T - 2                    # dy index 范围
    yy = dy[lo:hi + 1]                   # Δy_{t}
    ylag = y[lo:hi + 1]                  # y_{t-1} (dy index lo 对应 y[lo])
    cols = [dy[lo - j:hi + 1 - j] for j in range(1, p + 1)]   # Δy_{t-j}
    X = np.column_stack([np.ones(len(yy)), ylag, *cols])
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    e = yy - X @ b
    s2 = np.sum(e ** 2) / (X.shape[0] - X.shape[1])
    se_g = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return b[1] / se_g


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 600
    # 单位根
    rw = np.cumsum(rng.normal(0, 1, T))
    # 平稳 AR(2)
    st = np.zeros(T)
    for t in range(2, T):
        st[t] = 0.6 * st[t - 1] - 0.1 * st[t - 2] + rng.normal()
    t_ur = adf_stat(rw)
    t_st = adf_stat(st)
    print(f"单位根   ADF t = {t_ur:.2f} (应不显著, 接受单位根)")
    print(f"平稳AR(2)ADF t = {t_st:.2f} (应显著, 拒绝单位根)")
    print("DF 5% 临界(含常数)≈ -2.86")
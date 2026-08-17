# 03_20.4 单位根检验综述———差分还是不差分?

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def adf(y):
    T = len(y); dy = np.diff(y)
    X = np.column_stack([np.ones(T - 1), y[:-1]])
    b = np.linalg.lstsq(X, dy, rcond=None)[0]
    e = dy - X @ b
    s2 = np.sum(e ** 2) / (T - 3)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return b[1] / se


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 500
    st = np.zeros(T)
    for t in range(1, T):
        st[t] = 0.6 * st[t - 1] + rng.normal()    # 平稳
    rw = np.cumsum(rng.normal(0, 1, T))           # 单位根
    crit = -2.86

    t_st = adf(st); t_rw = adf(rw)
    def decide(t):
        return "不差分(水平)" if t < crit else "差分"
    print(f"平稳 AR:    ADF={t_st:.2f} -> {decide(t_st)}")
    print(f"单位根:    ADF={t_rw:.2f} -> {decide(t_rw)}")
    print("协整时用 VECM/误差修正(见 20.2)")
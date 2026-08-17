# 04_第19章习题

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def eg_adf(y1, y2):
    T = len(y1)
    X = np.column_stack([np.ones(T), y2])
    b = np.linalg.lstsq(X, y1, rcond=None)[0]
    u = y1 - X @ b
    dy = np.diff(u); Xa = u[:-1]
    g = np.dot(Xa, dy) / np.dot(Xa, Xa)
    e = dy - g * Xa
    s2 = np.sum(e ** 2) / (len(u) - 2)
    se = np.sqrt(s2 / np.dot(Xa, Xa))
    return g / se


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 3000
    y2 = np.cumsum(rng.normal(0, 1, T))
    y1c = 1.5 * y2 + rng.normal(0, 0.5, T)      # 协整
    x1 = np.cumsum(rng.normal(0, 1, T)); x2 = np.cumsum(rng.normal(0, 1, T))
    print("习题1/2: EG 检验")
    print("  协整对 t =", round(eg_adf(y1c, y2), 2), " (拒绝无协整)")
    print("  独立对 t =", round(eg_adf(x1, x2), 2), " (不能拒绝)")
    print("习题3: 协整系数估计 SD~1/T(第19.A 已验证)")
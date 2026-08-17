# 01_19.2 零假设为没有协整关系的检验

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def eg_adf(y1, y2):
    """EG: 回归 y1~y2, 对残差做 ADF, 返回残差 ADF t。"""
    T = len(y1)
    X = np.column_stack([np.ones(T), y2])
    b = np.linalg.lstsq(X, y1, rcond=None)[0]
    u = y1 - X @ b
    # 对 u 做 ADF(1 滞后) 无常数: Δu = g u_{t-1} + ... 
    T2 = len(u) - 1
    dy = np.diff(u)
    Xa = u[:-1]
    g = np.dot(Xa, dy) / np.dot(Xa, Xa)
    e = dy - g * Xa
    s2 = np.sum(e ** 2) / (T2 - 1)
    se = np.sqrt(s2 / np.dot(Xa, Xa))
    return g / se


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 800
    # 协整对
    y2 = np.cumsum(rng.normal(0, 1, T))
    y1c = 1.5 * y2 + rng.normal(0, 0.5, T)
    # 独立对(无协整)
    x1 = np.cumsum(rng.normal(0, 1, T))
    x2 = np.cumsum(rng.normal(0, 1, T))
    t_co = eg_adf(y1c, y2)
    t_ind = eg_adf(x1, x2)
    print(f"协整对 EG-ADF t = {t_co:.2f}  (应极负, 拒绝无协整)")
    print(f"独立对 EG-ADF t = {t_ind:.2f}  (应温和, 不能拒绝无协整)")
    print("EG 5% 临界(2 变量)约 -3.3")
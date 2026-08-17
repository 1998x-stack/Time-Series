# 04_4.5 线性投影更新

"""
Lecture: /第4章 预测
Content: 04_4.5 线性投影更新
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def proj(y, X):
    """y 对 X (含常数列) 的 OLS 投影, 返回预测值与残差。"""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    pred = X1 @ beta
    return beta, pred, y - pred


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 100000
    # 生成三元高斯 (X1,X2,Y) 用含噪声的线性结构
    X1 = rng.normal(size=n)
    X2 = 0.5 * X1 + rng.normal(size=n)
    Y = 1.0 * X1 + 0.7 * X2 + rng.normal(size=n)

    # 1) 基于旧信息 X2 的预测
    b0, pred0, res0 = proj(Y, X2[:, None])

    # 2) 新息: X1 未被 X2 解释的部分
    _, px1, nu = proj(X1, X2[:, None])

    # 3) 修正系数 b = Cov(Y-res0, nu)/Var(nu)
    b = np.cov(res0, nu)[0, 1] / np.var(nu)
    updated = pred0 + b * nu

    # 4) 联合投影 Y ~ [X1, X2]
    _, pred_full, _ = proj(Y, np.column_stack([X1, X2]))

    print("修正系数 b =", round(float(b), 4))
    print("更新预测 vs 联合投影 max-diff:", np.max(np.abs(updated - pred_full)))
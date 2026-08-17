# 06_4.7 自回归移动平均过程的和

"""
Lecture: /第4章 预测
Content: 06_4.7 自回归移动平均过程的和
"""

import numpy as np


def sample_acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    n = len(x)
    xc = x - x.mean()
    return np.array([np.dot(xc[h:], xc[: n - h]) / n for h in range(maxlag + 1)])


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi = 0.7
    sig_s = 1.0
    sig_w = 0.5
    n = 20000

    # 模拟 AR(1) 信号 s
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0.0, sig_s)
    w = rng.normal(0.0, sig_w, n)
    y = s + w

    # 理论自协方差
    sigs2 = sig_s ** 2 / (1 - phi ** 2)
    h = np.arange(0, 7)
    theory = np.where(h == 0, sigs2 + sig_w ** 2, sigs2 * phi ** h)

    ghat = sample_acf(y, 6)
    print("样本 γ_y:", np.round(ghat, 3))
    print("理论 γ_y:", np.round(theory, 3), " (h=0 被噪声抬高)")

    # 自相关(rho) h>=1 处以 phi^h 衰减, 说明 AR(1) 主导
    rho = ghat / ghat[0]
    print("样本自相关 φ^h 对照: φ^1..φ^6 =", np.round(phi ** np.arange(1, 7), 3))
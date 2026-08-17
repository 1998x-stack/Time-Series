# 03_4.4 正定对称矩阵的三角分解

"""
Lecture: /第4章 预测
Content: 03_4.4 正定对称矩阵的三角分解
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


def toeplitz_acf(gamma_0_to_p, n):
    """由 gamma(0..p) 构造 Toeplitz 协方差矩阵, gamma(h)=gamma(|i-j|)。"""
    # 先延拓: 对 AR(p) 用递推, 此处输入完整 gamma 序列长度 n 更简单
    # 这里仅对输入末尾到 n 的 gamma 做 Toeplitz
    if len(gamma_0_to_p) < n:
        raise ValueError("需要提供长度 >= n 的 gamma")
    g = gamma_0_to_p[:n]
    m = np.fromfunction(lambda i, j: np.abs(i - j), (n, n), dtype=int)
    return g[m]


def ar1_acf(phi: float, sigma: float, n: int) -> np.ndarray:
    """AR(1) 理论自协方差 gamma(h)=sigma^2 phi^h/(1-phi^2)。"""
    h = np.arange(n)
    return sigma ** 2 * phi ** h / (1 - phi ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi = 0.7
    sigma = 1.0
    n = 8
    gamma = ar1_acf(phi, sigma, n)
    G = toeplitz_acf(gamma, n)

    L = np.linalg.cholesky(G)
    print("Cholesky 重建误差 ||LL' - Γ|| =", np.max(np.abs(L @ L.T - G)))

    # 用 L 从白噪声重造 AR(1): x = L v  ⇒ Cov(x)=LL'=Γ (统计验证)
    B = 4000
    V = rng.normal(size=(n, B))
    X = L @ V
    Ghat = np.cov(X)
    print("模拟 Cov(x) 与 Γ 最大差:", np.max(np.abs(Ghat - G)))

    # 白化 X: L^{-1} X 协方差 ≈ I
    W = np.linalg.solve(L, X)
    Ghat_w = np.cov(W)
    print("L^{-1}x 的协方差应≈I, 与 I 最大差:", np.max(np.abs(Ghat_w - np.eye(n))))
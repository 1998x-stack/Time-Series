# 00_13.1 动态系统的状态空间表示

import numpy as np


def simulate_state_space(F, Q, H, V, T, rng):
    """模拟: xi_t = F xi_{t-1} + v_t; y_t = H xi_t + w_t。返回 (Xi, Y)。"""
    n = F.shape[0]
    LQ = np.linalg.cholesky(Q); LV = np.linalg.cholesky(V)
    xi = np.zeros((T, n)); y = np.zeros(T)
    for t in range(T):
        if t > 0:
            xi[t] = F @ xi[t - 1] + LQ @ rng.normal(size=n)
        y[t] = (H @ xi[t]).item() + (LV @ rng.normal(size=V.shape[0]))[0]
    return xi, y


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    # AR(1) 潜状态 + 观测噪声: F=[phi], Q=[q], H=[1], V=[r]
    phi, q, r = 0.8, 1.0, 0.5
    xi, y = simulate_state_space(np.array([[phi]]), np.array([[1.0]]),
                                 np.array([[1.0]]), np.array([[r]]), 200, rng)
    xi1 = xi[:, 0]
    print("潜状态 ξ 标准差:", round(xi1.std(), 3), "  潜状态自相关(h=1):",
          round(np.corrcoef(xi1[1:], xi1[:-1])[0, 1], 3))
    print("观测 y 与潜状态之差(噪声 r):", round(np.mean((y - xi[:, 0]) ** 2), 3),
          " ~ ", r)

    print("状态方程系数 OLS ξt~ξ_{t-1}:",
          round(np.dot(xi1[1:], xi1[:-1]) / np.dot(xi1[:-1], xi1[:-1]), 3),
          " ≈ phi =", phi)
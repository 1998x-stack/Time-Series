# 06_第10章习题

"""
Lecture: /第10章 协方差平稳的向量过程
Content: 06_第10章习题
"""

import numpy as np


if __name__ == "__main__":
    # 习题1: 平稳性
    Phi = np.array([[0.6, 0.2], [0.1, 0.5]])
    ev = np.linalg.eigvals(Phi)
    print(f"习题1: 特征值={np.round(ev,3)} 模={np.round(np.abs(ev),3)} 平稳={bool(np.max(np.abs(ev))<1)}")

    # 习题3: 长程方差 = 2πF(0)
    Omega = np.eye(2)
    Lam = np.linalg.inv(np.eye(2) - Phi) @ Omega @ np.linalg.inv(np.eye(2) - Phi.T)
    print("习题3: 长程方差 = (I-Φ)^{-1}Ω(I-Φ')^{-1} 对角线:",
          np.round(np.diag(Lam), 3), " = 2πF(0) 对角")
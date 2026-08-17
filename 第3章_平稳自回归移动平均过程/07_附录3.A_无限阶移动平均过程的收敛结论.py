# 07_附录3.A 无限阶移动平均过程的收敛结论

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 07_附录3.A 无限阶移动平均过程的收敛结论
"""

import numpy as np


def power_psi(beta: float, Jmax: int) -> np.ndarray:
    """psi_j = (j+1)^(-beta), j=0..Jmax。"""
    return (np.arange(1, Jmax + 2) ** (-beta))


def report(name: str, psi: np.ndarray, Js=(10, 100, 1000)):
    sq = np.cumsum(psi ** 2)
    ab = np.cumsum(np.abs(psi))
    print(name)
    for J in Js:
        print(f"  J={J:5d}  平方和={sq[J]:9.4f}  绝对值={ab[J]:9.4f}")
    print()


if __name__ == "__main__":
    Jmax = 1000
    # 几何衰减 psi_j = 0.8^j: 平方和与绝对和均收敛
    geom = 0.8 ** np.arange(Jmax + 1)
    report("psi_j = 0.8^j (几何, β 无穷)", geom)

    # 幂律 psi_j=(j+1)^-0.5: 平方和发散 (~log J)
    report("psi_j = (j+1)^-0.5 (β=1/2)", power_psi(0.5, Jmax))

    # 幂律 psi_j=(j+1)^-0.75: 平方和收敛, 绝对和发散
    report("psi_j = (j+1)^-0.75 (β=3/4)", power_psi(0.75, Jmax))
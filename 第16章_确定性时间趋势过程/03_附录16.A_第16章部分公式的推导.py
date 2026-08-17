# 03_附录16.A 第16章部分公式的推导

import numpy as np


if __name__ == "__main__":
    for T in (100, 1000, 10000):
        t = np.arange(1, T + 1).astype(float)
        s2 = np.sum(t ** 2) / T ** 3
        s1 = np.sum(t / T) / T
        print(f"T={T:6d}: (1/T³)Σt²={s2:.5f} (→1/3={1/3:.5f}), "
              f"(1/T)Σ(t/T)={s1:.5f} (→1/2)")
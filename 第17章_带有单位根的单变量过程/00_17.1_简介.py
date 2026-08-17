# 00_17.1 简介

import numpy as np


def sim_ar(phi, n, rng):
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal()
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 2000
    rw = np.cumsum(rng.normal(0, 1, T))          # 单位根
    st = sim_ar(0.8, T, rng)                     # 平稳
    print("随机游走(单位根): 末段(后500)方差", round(rw[-500:].var(), 2))
    print("平稳 AR(0.8):     末段方差          ", round(st[-500:].var(), 2))
    print("单位根方差随样本段增长 vs 平稳稳定 => 本质区别")
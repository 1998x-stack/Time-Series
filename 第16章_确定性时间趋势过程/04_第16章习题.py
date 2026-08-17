# 04_第16章习题

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    # 习题1: Var(delta_hat) = 12 sigma^2 / T^3
    T, sig2 = 400, 1.0
    var_d = 12 * sig2 / T ** 3
    print(f"习题1: Var(δ̂) = 12σ²/T³ = {var_d:.3e}")

    # 习题2: t 检验近似标准正态 (对照 16.2)
    print("习题2: δ 的 t 统计在平稳误差下近似 N(0,1)")

    # 习题3: DR/趋势对比
    print("习题3: 趋势δ的t统计-标准正态; 单位根 ADF 检验用DF分布")
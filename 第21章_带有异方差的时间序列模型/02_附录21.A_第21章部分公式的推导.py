# 02_附录21.A 第21章部分公式的推导

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*matmul")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    omega, alpha = 0.2, 0.4     # 3α²=0.48<1
    T = 200000
    h = np.zeros(T); y = np.zeros(T)
    for t in range(1, T):
        h[t] = omega + alpha * y[t - 1] ** 2
        y[t] = np.sqrt(h[t]) * rng.normal()

    var_theo = omega / (1 - alpha)
    kurt_theo = 3 * (1 - alpha ** 2) / (1 - 3 * alpha ** 2)
    kurt_s = np.mean(((y - y.mean()) / y.std()) ** 4)
    print(f"ARCH(1) α={alpha}")
    print(f"  方差: 样本={y.var():.3f} 理论={var_theo:.3f}")
    print(f"  峰度: 样本={kurt_s:.3f}  理论={kurt_theo:.3f} (厚尾)")
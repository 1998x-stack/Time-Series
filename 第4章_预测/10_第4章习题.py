# 10_第4章习题

"""
Lecture: /第4章 预测
Content: 10_第4章习题
"""

import numpy as np


def forecast_ar1(phi, sigma, xT, h):
    """AR(1) h 步预测与 MSE。"""
    fc = phi ** h * xT
    mse = sigma ** 2 * (1 - phi ** (2 * h)) / (1 - phi ** 2)
    return fc, mse


if __name__ == "__main__":
    # 习题1: AR(1) 0.7, xT=2
    phi, sigma, xT = 0.7, 1.0, 2.0
    for h in (1, 2):
        fc, mse = forecast_ar1(phi, sigma, xT, h)
        print(f"习题1 h={h}: 预测={fc:.3f}  MSE={mse:.3f}")

    # 习题2: MA(1) forecast = theta*eps_T
    theta, epsT = 0.5, 0.3
    print(f"习题2: 预测 x_{{T+1}} = theta*eps_T = {theta*epsT:.2f} (非0)")

    # 习题3: 线性投影系数与 MSE
    vx, vy, cxy = 1.0, 2.0, 0.8
    beta = cxy / vx
    mse3 = vy - beta * cxy
    print(f"习题3: beta={beta:.2f}  MSE={mse3:.2f} (=VarY - Cov^2/VarX)")
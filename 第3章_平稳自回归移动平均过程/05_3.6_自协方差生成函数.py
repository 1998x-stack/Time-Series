# 05_3.6 自协方差生成函数

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 05_3.6 自协方差生成函数
"""


import numpy as np
import matplotlib.pyplot as plt

# 定义 MA(1) 过程的参数
theta = 0.5
sigma2 = 1.0

# 定义自协方差生成函数
def autocovariance_generating_function(z, theta, sigma2):
    return sigma2 * (1 + theta * z) * (1 + theta * z**-1)

# 在单位圆上取点
theta_values = np.linspace(0, 2 * np.pi, 500)
z_values = np.exp(1j * theta_values)  # 复数单位圆上的点

# 计算自协方差生成函数的值
gamma_values = [autocovariance_generating_function(z, theta, sigma2) for z in z_values]

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(theta_values, np.abs(gamma_values))
plt.title('Autocovariance Generating Function on the Unit Circle for MA(1) Process')
plt.xlabel('Angle (radians)')
plt.ylabel('Magnitude of Gamma(z)')
plt.grid(True)
plt.show()

"""
有限模：从图中可以看出，自协方差生成函数在单位圆上的模是有限的，这表明该时间序列的自协方差在所有滞后下都是有限的，符合平稳性的要求。
平稳性判定：通过自协方差生成函数在单位圆上的分析，可以有效地判断时间序列是否平稳。对于MA(1)过程，如果自协方差生成函数在单位圆上的模是有限的，则该过程是平稳的。
"""
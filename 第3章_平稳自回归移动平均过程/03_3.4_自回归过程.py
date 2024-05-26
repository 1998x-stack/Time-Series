# 03_3.4 自回归过程

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 03_3.4 自回归过程
"""

import numpy as np
from typing import List

class AutoregressiveProcess:
    """
    自回归过程（AR）类，用于模拟和分析 AR(p) 模型。

    Attributes:
        order (int): 自回归过程的阶数 p。
        constant (float): 常数项 c。
        ar_params (np.ndarray): 自回归参数 φ。
        noise_variance (float): 白噪声的方差 σ^2。
    """

    def __init__(self, order: int, constant: float, ar_params: List[float], noise_variance: float):
        """
        初始化 AR 类的实例。

        Args:
            order (int): 自回归过程的阶数 p。
            constant (float): 常数项 c。
            ar_params (List[float]): 自回归参数 φ。
            noise_variance (float): 白噪声的方差 σ^2。
        """
        self.order = order
        self.constant = constant
        self.ar_params = np.array(ar_params)
        self.noise_variance = noise_variance
        self.noise_mean = 0  # 白噪声均值为0

    def __repr__(self) -> str:
        return (f"AutoregressiveProcess(order={self.order}, constant={self.constant}, "
                f"ar_params={self.ar_params.tolist()}, noise_variance={self.noise_variance})")

    def generate_series(self, n: int) -> np.ndarray:
        """
        生成 AR(p) 过程的时间序列。

        Args:
            n (int): 生成的时间序列长度。

        Returns:
            np.ndarray: 生成的时间序列。
        """
        noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_variance), n + self.order)
        series = np.zeros(n + self.order)
        
        for t in range(self.order, n + self.order):
            series[t] = self.constant + noise[t]
            for j in range(1, self.order + 1):
                series[t] += self.ar_params[j-1] * series[t - j]
        
        return series[self.order:]

    def calculate_expectation(self) -> float:
        """
        计算 AR(p) 过程的期望值。

        Returns:
            float: 期望值。
        """
        return self.constant / (1 - np.sum(self.ar_params))

    def calculate_variance(self) -> float:
        """
        计算 AR(p) 过程的方差。

        Returns:
            float: 方差。
        """
        denominator = 1 - np.sum(self.ar_params ** 2)
        if denominator <= 0:
            raise ValueError("方差计算中出现非正值，请检查AR参数的平稳性。")
        variance = self.noise_variance / denominator
        return variance

    def calculate_autocorrelation(self, lag: int) -> float:
        """
        计算 AR(p) 过程的自相关函数。

        Args:
            lag (int): 滞后阶数。

        Returns:
            float: 滞后 lag 的自相关值。
        """
        if lag == 0:
            return 1.0
        if lag > self.order:
            return 0.0
        
        autocorrelation = self.ar_params[lag-1]
        return autocorrelation

# 实例化AR(1)模型
ar1 = AutoregressiveProcess(order=1, constant=0, ar_params=[0.5], noise_variance=1.0)
print(ar1)

# 生成时间序列
series = ar1.generate_series(100)
print("Generated Series:", series[:10])  # 仅显示前10个数据点

# 计算期望值
expectation = ar1.calculate_expectation()
print("Expectation:", expectation)

# 计算方差
variance = ar1.calculate_variance()
print("Variance:", variance)

# 计算自相关函数
autocorrelation_0 = ar1.calculate_autocorrelation(0)
autocorrelation_1 = ar1.calculate_autocorrelation(1)
autocorrelation_2 = ar1.calculate_autocorrelation(2)
print("Autocorrelation (lag 0):", autocorrelation_0)
print("Autocorrelation (lag 1):", autocorrelation_1)
print("Autocorrelation (lag 2):", autocorrelation_2)

# 可视化时间序列
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(series, label='AR(1) Series')
plt.title('Generated AR(1) Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 实例化AR(1)模型
ar1 = AutoregressiveProcess(order=1, constant=0, ar_params=[0.5], noise_variance=1.0)
print(ar1)

# 生成时间序列
series = ar1.generate_series(100)
print("Generated Series:", series[:10])  # 仅显示前10个数据点

# 计算期望值
expectation = ar1.calculate_expectation()
print("Expectation:", expectation)

# 计算方差
variance = ar1.calculate_variance()
print("Variance:", variance)

# 计算自相关函数
autocorrelation_0 = ar1.calculate_autocorrelation(0)
print("Autocorrelation (lag 0):", autocorrelation_0)
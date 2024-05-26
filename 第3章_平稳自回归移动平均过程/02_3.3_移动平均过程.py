import numpy as np
import matplotlib.pyplot as plt
from typing import List

class MovingAverageProcess:
    """
    移动平均过程（MA）类，用于模拟和分析 MA(q) 模型。

    Attributes:
        order (int): 移动平均过程的阶数 q。
        mean (float): 序列的均值 μ。
        ma_params (np.ndarray): 移动平均参数 θ。
        noise_variance (float): 白噪声的方差 σ^2。
    """

    def __init__(self, order: int, mean: float, ma_params: List[float], noise_variance: float):
        """
        初始化 MA 类的实例。

        Args:
            order (int): 移动平均过程的阶数 q。
            mean (float): 序列的均值 μ。
            ma_params (List[float]): 移动平均参数 θ。
            noise_variance (float): 白噪声的方差 σ^2。
        """
        self.order = order
        self.mean = mean
        self.ma_params = np.array(ma_params)
        self.noise_variance = noise_variance
        self.noise_mean = 0  # 白噪声均值为0

    def __repr__(self) -> str:
        return (f"MovingAverageProcess(order={self.order}, mean={self.mean}, "
                f"ma_params={self.ma_params.tolist()}, noise_variance={self.noise_variance})")

    def generate_series(self, n: int) -> np.ndarray:
        """
        生成 MA(q) 过程的时间序列。

        Args:
            n (int): 生成的时间序列长度。

        Returns:
            np.ndarray: 生成的时间序列。
        """
        noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_variance), n + self.order)
        series = np.zeros(n + self.order)
        
        for t in range(self.order, n + self.order):
            series[t] = self.mean + noise[t]
            for j in range(1, self.order + 1):
                series[t] += self.ma_params[j-1] * noise[t - j]
        
        return series[self.order:]

    def calculate_expectation(self) -> float:
        """
        计算 MA(q) 过程的期望值。

        Returns:
            float: 期望值。
        """
        return self.mean

    def calculate_variance(self) -> float:
        """
        计算 MA(q) 过程的方差。

        Returns:
            float: 方差。
        """
        variance = self.noise_variance * (1 + np.sum(self.ma_params ** 2))
        return variance

    def calculate_autocorrelation(self, lag: int) -> float:
        """
        计算 MA(q) 过程的自相关函数。

        Args:
            lag (int): 滞后阶数。

        Returns:
            float: 滞后 lag 的自相关值。
        """
        if lag > self.order:
            return 0.0
        
        autocorrelation = 0.0
        for j in range(self.order - lag):
            autocorrelation += self.ma_params[j] * self.ma_params[j + lag]
        
        return self.noise_variance * autocorrelation


# 实例化MA(1)模型
ma1 = MovingAverageProcess(order=1, mean=0, ma_params=[0.5], noise_variance=1.0)
print(ma1)

# 生成时间序列
series = ma1.generate_series(100)
print("Generated Series:", series[:10])  # 仅显示前10个数据点

# 计算期望值
expectation = ma1.calculate_expectation()
print("Expectation:", expectation)

# 计算方差
variance = ma1.calculate_variance()
print("Variance:", variance)

# 计算自相关函数
autocorrelation_0 = ma1.calculate_autocorrelation(0)
autocorrelation_1 = ma1.calculate_autocorrelation(1)
autocorrelation_2 = ma1.calculate_autocorrelation(2)
print("Autocorrelation (lag 0):", autocorrelation_0)
print("Autocorrelation (lag 1):", autocorrelation_1)
print("Autocorrelation (lag 2):", autocorrelation_2)

# 可视化时间序列
plt.figure(figsize=(10, 6))
plt.plot(series, label='MA(1) Series')
plt.title('Generated MA(1) Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

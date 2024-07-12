import numpy as np
from typing import Tuple

class DurbinLevinson:
    def __init__(self, autocovariances: np.ndarray):
        """
        初始化Durbin-Levinson算法类。

        参数:
        autocovariances (np.ndarray): 自协方差函数值的数组，长度为 p+1，其中 p 为最大滞后阶数。
        """
        self.autocovariances = autocovariances
        self.p = len(autocovariances) - 1
        self.phi = np.zeros((self.p + 1, self.p + 1))
        self.sigma = np.zeros(self.p + 1)
        self.calculate_parameters()

    def calculate_parameters(self):
        """
        使用Durbin-Levinson算法计算AR(p)模型的参数。
        """
        # 初始条件
        self.phi[1, 1] = self.autocovariances[1] / self.autocovariances[0]
        self.sigma[1] = self.autocovariances[0] * (1 - self.phi[1, 1] ** 2)
        
        # 递归计算
        for k in range(2, self.p + 1):
            phi_sum = sum(self.phi[k - 1, j] * self.autocovariances[k - j] for j in range(1, k))
            self.phi[k, k] = (self.autocovariances[k] - phi_sum) / self.sigma[k - 1]
            
            for j in range(1, k):
                self.phi[k, j] = self.phi[k - 1, j] - self.phi[k, k] * self.phi[k - 1, k - j]
            
            self.sigma[k] = self.sigma[k - 1] * (1 - self.phi[k, k] ** 2)

    def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取AR(p)模型的参数和噪声方差。

        返回:
        Tuple[np.ndarray, np.ndarray]: 返回两个数组，第一个是AR(p)模型的参数，第二个是噪声方差。
        """
        coefficients = self.phi[self.p, 1:self.p + 1]
        noise_variance = self.sigma[self.p]
        return coefficients, noise_variance

def example_usage():
    """
    示例用法：使用Durbin-Levinson算法计算AR模型的参数和噪声方差。
    """
    # 示例自协方差值
    autocovariances = np.array([1.0, 0.5, 0.3, 0.2, 0.1])

    # 初始化Durbin-Levinson算法类
    dl = DurbinLevinson(autocovariances)

    # 获取AR模型的参数和噪声方差
    coefficients, noise_variance = dl.get_coefficients()

    # 打印结果
    print("AR模型的参数:", coefficients)
    print("噪声方差:", noise_variance)

# 执行示例用法
if __name__ == "__main__":
    example_usage()

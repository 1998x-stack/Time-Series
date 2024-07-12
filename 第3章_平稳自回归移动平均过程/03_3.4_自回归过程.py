# 03_3.4 自回归过程

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 03_3.4 自回归过程
"""

import numpy as np
from typing import List

class AutoregressiveProcess:
    """
    自回归过程类，用于生成和分析AR(p)模型。

    Attributes:
        coefficients (List[float]): AR模型的参数列表
        order (int): AR模型的阶数
    """

    def __init__(self, coefficients: List[float]):
        """
        初始化自回归过程。

        Args:
            coefficients (List[float]): AR模型的参数列表
        """
        self.coefficients = np.array(coefficients)
        self.order = len(coefficients)
    
    def generate_samples(self, n_samples: int, noise_variance: float = 1.0) -> np.ndarray:
        """
        生成自回归过程的样本序列。

        Args:
            n_samples (int): 生成样本的数量
            noise_variance (float): 噪声方差，默认值为1.0

        Returns:
            np.ndarray: 生成的AR过程样本序列
        """
        # 初始化样本序列
        samples = np.zeros(n_samples)
        # 生成白噪声序列
        white_noise = np.random.normal(scale=np.sqrt(noise_variance), size=n_samples)
        
        # 生成AR过程样本
        for t in range(self.order, n_samples):
            samples[t] = np.dot(self.coefficients, samples[t-self.order:t][::-1]) + white_noise[t]
        
        return samples

    def estimate_parameters(self, samples: np.ndarray) -> np.ndarray:
        """
        使用Yule-Walker方程估计AR模型参数。

        Args:
            samples (np.ndarray): 样本序列

        Returns:
            np.ndarray: 估计的AR模型参数
        """
        from scipy.linalg import toeplitz

        # 计算自相关函数
        r = np.correlate(samples, samples, mode='full')[len(samples)-1:]
        r = r[:self.order+1]
        
        # 构建Toeplitz矩阵
        R = toeplitz(r[:-1])
        r = r[1:]
        
        # 计算AR模型参数
        phi_hat = np.linalg.solve(R, r)
        
        return phi_hat

def main():
    """
    主函数，演示自回归过程的使用。
    """
    # 定义AR(2)模型的参数
    coefficients = [0.75, -0.25]
    ar_process = AutoregressiveProcess(coefficients)
    
    # 生成AR过程样本
    n_samples = 100
    noise_variance = 1.0
    samples = ar_process.generate_samples(n_samples, noise_variance)
    
    # 打印生成的样本序列的前10个值
    print("Generated AR process samples (first 10 samples):")
    print(samples[:10])
    
    # 使用Yule-Walker方程估计AR模型参数
    estimated_coefficients = ar_process.estimate_parameters(samples)
    
    # 打印估计的参数
    print("Estimated AR process coefficients:")
    print(estimated_coefficients)

if __name__ == "__main__":
    main()
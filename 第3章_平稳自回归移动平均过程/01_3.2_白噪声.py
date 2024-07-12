# 01_3.2 白噪声

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 01_3.2 白噪声
"""

import numpy as np

class WhiteNoiseGenerator:
    """ 
    白噪声生成器类，生成具有特定均值和方差的白噪声序列。

    Attributes:
        mean (float): 白噪声的均值
        variance (float): 白噪声的方差
        length (int): 序列的长度
    """

    def __init__(self, mean: float = 0, variance: float = 1, length: int = 1000):
        """
        初始化白噪声生成器。

        Args:
            mean (float): 白噪声的均值，默认值为0
            variance (float): 白噪声的方差，默认值为1
            length (int): 序列的长度，默认值为1000
        """
        self.mean = mean
        self.variance = variance
        self.length = length

    def generate(self) -> np.ndarray:
        """
        生成白噪声序列。

        Returns:
            np.ndarray: 生成的白噪声序列
        """
        # 生成符合正态分布的随机序列
        white_noise = np.random.normal(self.mean, np.sqrt(self.variance), self.length)
        return white_noise

    def compute_power_spectral_density(self, white_noise: np.ndarray) -> np.ndarray:
        """
        计算白噪声序列的功率谱密度。

        Args:
            white_noise (np.ndarray): 白噪声序列

        Returns:
            np.ndarray: 功率谱密度
        """
        # 计算傅里叶变换
        fft_result = np.fft.fft(white_noise)
        # 计算功率谱密度
        psd = (np.abs(fft_result) ** 2) / len(white_noise)
        return psd

def main():
    """
    主函数，演示白噪声生成器的使用。
    """
    # 初始化白噪声生成器
    mean = 0
    variance = 1
    length = 1000

    white_noise_generator = WhiteNoiseGenerator(mean, variance, length)

    # 生成白噪声序列
    white_noise = white_noise_generator.generate()

    # 计算功率谱密度
    psd = white_noise_generator.compute_power_spectral_density(white_noise)

    # 打印结果
    print(f"Generated White Noise (first 10 samples): {white_noise[:10]}")
    print(f"Power Spectral Density (first 10 values): {psd[:10]}")

if __name__ == "__main__":
    main()
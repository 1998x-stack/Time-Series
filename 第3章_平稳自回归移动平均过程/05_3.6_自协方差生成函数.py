# 05_3.6 自协方差生成函数

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 05_3.6 自协方差生成函数
"""

import numpy as np
from typing import List

class AutocovarianceGeneratingFunction:
    """
    自协方差生成函数 (ACGF) 类，用于计算和分析时间序列的自协方差生成函数。

    Attributes:
        autocovariances (List[float]): 时间序列的自协方差函数值列表
    """

    def __init__(self, autocovariances: List[float]):
        """
        初始化自协方差生成函数。

        Args:
            autocovariances (List[float]): 时间序列的自协方差函数值列表
        """
        self.autocovariances = np.array(autocovariances)
    
    def compute_acgf(self, z: complex) -> complex:
        """
        计算自协方差生成函数的值。

        Args:
            z (complex): 复数变量

        Returns:
            complex: 自协方差生成函数在 z 处的值
        """
        acgf_value = np.sum(self.autocovariances * np.array([z**k for k in range(-len(self.autocovariances) + 1, len(self.autocovariances))]))
        return acgf_value

    def compute_acgf_series(self, z_values: List[complex]) -> List[complex]:
        """
        计算一系列 z 值对应的自协方差生成函数的值。

        Args:
            z_values (List[complex]): 复数变量列表

        Returns:
            List[complex]: 自协方差生成函数在 z_values 处的值列表
        """
        return [self.compute_acgf(z) for z in z_values]

def main():
    """
    主函数，演示自协方差生成函数的使用。
    """
    # 示例自协方差函数值列表
    autocovariances = [1, 0.75, 0.5, 0.25, 0]
    
    # 创建自协方差生成函数对象
    acgf = AutocovarianceGeneratingFunction(autocovariances)
    
    # 计算并打印自协方差生成函数在一些复数点的值
    z_values = [0.5 + 0.5j, 1 + 0j, -0.5 + 0.5j]
    acgf_values = acgf.compute_acgf_series(z_values)
    
    for z, value in zip(z_values, acgf_values):
        print(f"ACGF at z = {z}: {value}")

if __name__ == "__main__":
    main()
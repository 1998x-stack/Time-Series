# 06_3.7 可逆性

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 06_3.7 可逆性
"""

import numpy as np
from typing import List

class MAProcess:
    """
    移动平均过程 (MA) 类，用于生成和分析 MA(q) 模型的可逆性。

    Attributes:
        ma_params (List[float]): MA 模型的参数列表
    """

    def __init__(self, ma_params: List[float]):
        """
        初始化 MA 过程。

        Args:
            ma_params (List[float]): MA 模型的参数列表
        """
        self.ma_params = np.array(ma_params)
        self.q = len(ma_params)  # MA 模型的阶数
    
    def is_invertible(self) -> bool:
        """
        检查 MA 模型是否可逆。

        Returns:
            bool: 如果模型可逆，返回 True；否则返回 False
        """
        # 构建特征多项式
        ma_poly = np.concatenate(([1], self.ma_params))
        
        # 计算特征多项式的根
        roots = np.roots(ma_poly)
        
        # 检查所有根的模是否都大于 1
        is_invertible = np.all(np.abs(roots) > 1)
        return is_invertible

def main():
    """
    主函数，演示 MA 过程的可逆性检查。
    """
    # 定义 MA(1) 模型的参数
    ma_params = [0.65]
    ma_process = MAProcess(ma_params)
    
    # 检查模型的可逆性
    invertibility = ma_process.is_invertible()
    
    # 打印检查结果
    print(f"Is the MA model invertible? {invertibility}")

    # 另一个示例
    ma_params = [1.5]
    ma_process = MAProcess(ma_params)
    
    # 检查模型的可逆性
    invertibility = ma_process.is_invertible()
    
    # 打印检查结果
    print(f"Is the MA model invertible? {invertibility}")

if __name__ == "__main__":
    main()


import numpy as np
from typing import List

class MAProcess:
    """
    移动平均过程 (MA) 类，用于生成和分析 MA(q) 模型的可逆性。

    Attributes:
        ma_params (List[float]): MA 模型的参数列表
    """

    def __init__(self, ma_params: List[float]):
        """
        初始化 MA 过程。

        Args:
            ma_params (List[float]): MA 模型的参数列表
        """
        self.ma_params = np.array(ma_params)
        self.q = len(ma_params)  # MA 模型的阶数
    
    def is_invertible_rouche(self) -> bool:
        """
        使用 Rouché 定理检查 MA 模型是否可逆。

        Returns:
            bool: 如果模型可逆，返回 True；否则返回 False
        """
        # 构建特征多项式
        ma_poly = np.concatenate(([1], self.ma_params))
        
        # 计算特征多项式的根
        roots = np.roots(ma_poly)
        
        # 检查所有根的模是否都大于 1
        is_invertible = np.all(np.abs(roots) > 1)
        return is_invertible
    
    def is_invertible_schur(self) -> bool:
        """
        使用 Schur-Cohn 判据检查 MA 模型是否可逆。

        Returns:
            bool: 如果模型可逆，返回 True；否则返回 False
        """
        a = np.array([1] + self.ma_params.tolist())
        n = len(a) - 1
        for k in range(n, 0, -1):
            if np.abs(a[-1]) >= 1:
                return False
            a = (a[:-1] - a[-1] * a[-2::-1]) / (1 - a[-1] ** 2)
        return True

def main():
    """
    主函数，演示 MA 过程的可逆性检查。
    """
    # 定义 MA(1) 模型的参数
    ma_params = [0.65]
    ma_process = MAProcess(ma_params)
    
    # 使用 Rouché 定理检查模型的可逆性
    invertibility_rouche = ma_process.is_invertible_rouche()
    
    # 打印检查结果
    print(f"Is the MA model invertible using Rouché's theorem? {invertibility_rouche}")

    # 使用 Schur-Cohn 判据检查模型的可逆性
    invertibility_schur = ma_process.is_invertible_schur()
    
    # 打印检查结果
    print(f"Is the MA model invertible using Schur-Cohn criterion? {invertibility_schur}")

    # 另一个示例
    ma_params = [1.5]
    ma_process = MAProcess(ma_params)
    
    # 使用 Rouché 定理检查模型的可逆性
    invertibility_rouche = ma_process.is_invertible_rouche()
    
    # 打印检查结果
    print(f"Is the MA model invertible using Rouché's theorem? {invertibility_rouche}")

    # 使用 Schur-Cohn 判据检查模型的可逆性
    invertibility_schur = ma_process.is_invertible_schur()
    
    # 打印检查结果
    print(f"Is the MA model invertible using Schur-Cohn criterion? {invertibility_schur}")

if __name__ == "__main__":
    main()

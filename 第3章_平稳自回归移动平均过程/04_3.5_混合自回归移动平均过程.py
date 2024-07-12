# 04_3.5 混合自回归移动平均过程

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 04_3.5 混合自回归移动平均过程
"""

import numpy as np
from typing import List, Tuple

class ARMAProcess:
    """
    混合自回归移动平均过程 (ARMA) 类，用于生成和分析 ARMA(p, q) 模型。

    Attributes:
        ar_params (List[float]): AR 模型的参数列表
        ma_params (List[float]): MA 模型的参数列表
    """

    def __init__(self, ar_params: List[float], ma_params: List[float]):
        """
        初始化 ARMA 过程。

        Args:
            ar_params (List[float]): AR 模型的参数列表
            ma_params (List[float]): MA 模型的参数列表
        """
        self.ar_params = np.array(ar_params)
        self.ma_params = np.array(ma_params)
        self.p = len(ar_params)  # AR 模型的阶数
        self.q = len(ma_params)  # MA 模型的阶数
    
    def generate_samples(self, n_samples: int, noise_variance: float = 1.0) -> np.ndarray:
        """
        生成 ARMA 过程的样本序列。

        Args:
            n_samples (int): 生成样本的数量
            noise_variance (float): 噪声方差，默认值为 1.0

        Returns:
            np.ndarray: 生成的 ARMA 过程样本序列
        """
        # 初始化样本序列
        samples = np.zeros(n_samples)
        # 生成白噪声序列
        white_noise = np.random.normal(scale=np.sqrt(noise_variance), size=n_samples)
        
        # 生成 ARMA 过程样本
        for t in range(max(self.p, self.q), n_samples):
            ar_component = np.dot(self.ar_params, samples[t-self.p:t][::-1])
            ma_component = np.dot(self.ma_params, white_noise[t-self.q:t][::-1])
            samples[t] = ar_component + white_noise[t] + ma_component
        
        return samples

    def estimate_parameters(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用极大似然估计法估计 ARMA 模型参数。

        Args:
            samples (np.ndarray): 样本序列

        Returns:
            Tuple[np.ndarray, np.ndarray]: 估计的 AR 模型参数和 MA 模型参数
        """
        from statsmodels.tsa.arima.model import ARIMA
        
        # 使用 statsmodels 库估计 ARMA 模型参数
        model = ARIMA(samples, order=(self.p, 0, self.q))
        model_fit = model.fit()
        
        ar_params_estimated = model_fit.arparams
        ma_params_estimated = model_fit.maparams
        
        return ar_params_estimated, ma_params_estimated

    def predict_n(self, n: int, history: np.ndarray) -> np.ndarray:
        """
        预测未来 n 步的值，并在每一步后重新拟合模型并获取新参数。

        Args:
            n (int): 要预测的步数
            history (np.ndarray): 历史数据

        Returns:
            np.ndarray: 预测的未来 n 步的值
        """
        predictions = np.zeros(n)
        extended_history = np.append(history, predictions)
        
        for t in range(n):
            # 重新估计参数
            current_history = extended_history[:len(history) + t]
            self.ar_params, self.ma_params = self.estimate_parameters(current_history)
            
            if t < self.q:
                ma_component = np.dot(self.ma_params[:t], extended_history[-t-1:-1][::-1])
            else:
                ma_component = np.dot(self.ma_params, extended_history[-self.q-1:-1][::-1])
                
            if t < self.p:
                ar_component = np.dot(self.ar_params[:t], extended_history[-t-1:-1][::-1])
            else:
                ar_component = np.dot(self.ar_params, extended_history[-self.p-1:-1][::-1])
                
            predictions[t] = ar_component + ma_component
            extended_history[len(history) + t] = predictions[t]
        
        return predictions

def main():
    """
    主函数，演示混合自回归移动平均过程的使用。
    """
    # 定义 ARMA(2, 2) 模型的参数
    ar_params = [0.75, -0.25]
    ma_params = [0.65, 0.35]
    arma_process = ARMAProcess(ar_params, ma_params)
    
    # 生成 ARMA 过程样本
    n_samples = 100
    noise_variance = 1.0
    samples = arma_process.generate_samples(n_samples, noise_variance)
    
    # 打印生成的样本序列的前 10 个值
    print("Generated ARMA process samples (first 10 samples):")
    print(samples[:10])
    
    # 使用极大似然估计法估计 ARMA 模型参数
    estimated_ar_params, estimated_ma_params = arma_process.estimate_parameters(samples)
    
    # 打印估计的参数
    print("Estimated AR parameters:")
    print(estimated_ar_params)
    print("Estimated MA parameters:")
    print(estimated_ma_params)
    
    # 预测未来 10 步的值
    history = samples[-max(len(ar_params), len(ma_params)):]  # 取最后的历史数据
    predictions = arma_process.predict_n(10, history)
    
    # 打印预测值
    print("Predicted future values:")
    print(predictions)

if __name__ == "__main__":
    main()
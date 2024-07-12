# 02_3.3 移动平均过程

"""
Lecture: /第3章 平稳自回归移动平均过程
Content: 02_3.3 移动平均过程
"""

import numpy as np
from typing import List

class MovingAverageProcess:
    def __init__(self, mean: float, theta: List[float]):
        """
        Initialize the Moving Average Process.

        Parameters:
        mean (float): Mean of the process.
        theta (List[float]): List of coefficients for the MA process.
        """
        self.mean = mean
        self.theta = np.array(theta)
        self.q = len(theta)  # Order of the MA process
        
    def generate_samples(self, n_samples: int):
        """
        Generate samples from the Moving Average Process.

        Parameters:
        n_samples (int): Number of samples to generate.

        Returns:
        np.ndarray: Array of shape (n_samples,) containing MA process samples.
        """
        # Generate white noise samples
        white_noise = np.random.normal(size=n_samples)
        
        # Initialize array to store MA process samples
        samples = np.zeros(n_samples)
        
        for t in range(self.q, n_samples):
            # Calculate MA process value at time t
            samples[t] = self.mean + np.sum(self.theta * white_noise[t-self.q:t])
        
        return samples

# Example usage
if __name__ == "__main__":
    # Define parameters for the MA process
    mean = 0.0
    theta = [0.5, -0.3]  # Example coefficients for the MA process, e.g., MA(2) process
    
    # Create MA process object
    ma_process = MovingAverageProcess(mean, theta)
    
    # Generate MA process samples
    n_samples = 100
    ma_samples = ma_process.generate_samples(n_samples)
    
    # Print the generated samples
    print("Generated MA process samples:")
    print(ma_samples)

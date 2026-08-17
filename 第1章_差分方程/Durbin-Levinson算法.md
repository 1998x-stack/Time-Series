# Durbin-Levinson算法

"""
Lecture: /第1章 差分方程
Content: Durbin-Levinson算法
"""

### 第1章 差分方程
#### Durbin-Levinson 算法

#### 引言
要对 AR(p) 过程做最优线性预测(第 4 章), 需要解一组由自协方差构成的 Yule-Walker 线性方程。直接用高斯消去解 $p\times p$ 系统的代价为 $O(p^3)$; Durbin-Levinson(DL) 算法利用方程组的 Toeplitz 结构, 以 $O(p^2)$ 递推一次求得全部阶的预报系数, 并顺带给出预报误差方差。这为后面的预测章节提供了关键算法工具。

#### 定义与 Yule-Walker 方程
设零均值平稳过程有自协方差 $\gamma_k = E(y_t y_{t-k})$。对 AR(p) 预报, 系数 $\phi_{p,1},\dots,\phi_{p,p}$ 满足 Yule-Walker 方程:
$$ \sum_{j=1}^{p} \phi_{p,j}\,\gamma_{k-j} = \gamma_k, \qquad k=1,\dots,p $$
预报误差方差为 $\sigma_p^2 = \gamma_0 - \sum_{j=1}^{p}\phi_{p,j}\gamma_j$。

#### Durbin-Levinson 递推
记 $\phi_{k,j}$ 为第 $k$ 阶预报中滞后 $j$ 的系数, $\sigma_k^2$ 为 $k$ 阶预报误差方差。递推($k=1,\dots,p$)如下:
$$
\phi_{k,k} = \frac{\gamma_k - \sum_{j=1}^{k-1} \phi_{k-1,j}\,\gamma_{k-j}}{\sigma_{k-1}^2},\qquad
\phi_{k,j} = \phi_{k-1,j} - \phi_{k,k}\,\phi_{k-1,k-j},\quad j=1,\dots,k-1
$$
$$
\sigma_k^2 = \sigma_{k-1}^2\left(1 - \phi_{k,k}^2\right)
$$
初值 $\sigma_0^2=\gamma_0$。$\phi_{k,k}$ 即偏自相关系数。整过程仅 $O(p^2)$。

#### 计算方法
`.py` 实现上面的三重递推: 维护二维 `phi[k, j]` 数组与一维 `sigma` 数组; 外层循环 $k$, 先算分子再用更新关系回代; 最后返回第 $p$ 行系数作为 $AR(p)$ 系数。

#### 数值演示
脚本生成一个已知 AR(2) 过程 $\phi=(0.5,-0.2)$(噪声方差 1), 用长样本估计其样本自协方差 $\gamma_0,\gamma_1,\gamma_2$, 代入 Durbin-Levinson 递推, 复原系数。预期估计系数接近真值 $(0.5,-0.2)$, 噪声方差接近 1。

#### 小结
Durbin-Levinson 是自协方差→AR 系数的经典 $O(p^2)$ 递推, 是预测(第 4 章)与 AR 拟合的基础工具, 且天然给出偏自相关与预报误差方差。
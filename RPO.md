**ROBUST POLICY OPTIMIZATION IN DEEP REINFORCEMENT LEARNING**

### Introduction

针对连续动作空间的梯度策略算法，面临两个问题：

1. 表达问题：策略网络通常输出高斯分布的均值和方差来表征动作的分布（高维动作空间，每个维度一个高斯分布）。但这不是普适的，有的task都要考虑用高斯分布以外的方式更合适
2. 探索问题：训练过程中策略网络会逐步减少动作选择的方差，也就是让动作更加的确定，这样会阻碍探索，使得陷入次优解

我们的方法尝试对动作分布的均值加上一个随机扰动来解决上面的问题。结果：

1. 输出的动作分布为正态分布的时候，在很多控制任务和benchmark上，我们的方法相比改动前有更好的表现，尤其在高维环境中（动作空间/状态空间高维）
2. 比两个基于数据增强的RL方法（RAD、DRAC）效果也要好
3. 对于其他参数化分布，例如s Laplace and Gumbe分布，在很多任务下，我们的方法效果更好
4. 相比在损失函数里加上entropy正则的方法，我们的方法表现更好。entropy正则的方法，有些系数下可能导致更差的表现。

bison：标准正态分布的entropy是一个常数，大约2.05 bit。

效果如下：

![image-20250524105938075](img/image-20250524105938075.png)

### Preliminaries

介绍了MDP、RL、PPO等预备知识



我特意补充一下我没有了解过的RL相关的数据增强的方法。

RAD（**Reinforcement Learning with Augmented Data**）和 DrAC（**Data-regularized Actor-Critic**）是用于增强视觉输入类强化学习（如图像输入）中agent泛化能力的方法，它们通过**数据增强（data augmentation）**提高样本利用效率和泛化性能。具体思路：

1. RAD：在使用图像作为输入的RL中，直接对观测图像应用数据增强（如crop、flip、color jitter等），增强后的图像作为神经网络的输入。网络训练过程和标准RL（如SAC）相同。
2. DrAC：不仅增强图像输入，还引入一个正则项，鼓励网络在原图和增强图之间预测的一致性：正则项 = 原图和增强图在策略概率分布或价值估计上的差异，这样引导Actor和Critic在图像增强前后的概率和值比较一致。

### ROBUST POLICY OPTIMIZATION (RPO)

![image-20250524114852606](img/image-20250524114852606.png)

### Results

#### 与PPO的对比

看了下面的数据，不禁想问：PPO真的表现这么差吗？不怕人踢馆吗？

![image-20250524115257487](img/image-20250524115257487.png)



![image-20250524115425070](img/image-20250524115425070.png)

#### 与熵正则方法的对比

![image-20250524115710514](img/image-20250524115710514.png)

#### 其他分布下的效果

![image-20250524120139630](img/image-20250524120139630.png)

#### 与数据增强方法的对比

![image-20250524120331569](img/image-20250524120331569.png)



#### 随机噪音范围的对比

![image-20250524120848084](img/image-20250524120848084.png)

### Related Work

提到了Entropy Regularization、输出高斯分布的策略方法、数据增强、GAE、clipped objective

### Conclusion



### bison的实验


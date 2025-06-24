Conservative Q-Learning for Offline Reinforcement Learning

### 1、Introduction

现有的离线强化学习（Offline RL）方法直接应用标准的 off-policy RL 算法会导致性能很差，主要是因为：

- 会从分布外（out-of-distribution, OOD）动作上 bootstrap，带来错误的乐观估计（过高的 Q 值）。其他论文里叫extrapolation error，老生常谈的问题
- 以及过拟合问题。

这两个问题最终表现为：错误地高估了价值函数。

作者认为：如果我们能学到一个保守（conservative）的价值函数估计，也就是它给出的 Q 值是对真实 Q 值的下界（lower bound），那么就能有效缓解这种过估计的问题。

方法的核心是通过简单修改标准的 value-based RL 算法来最小化 Q 值（使 Q 值偏保守），并结合一个基于数据分布的最大化项来收紧这个下界。

原文有一小段不好理解，特意问了AI：

![image-20250624151848959](img/image-20250624151848959.png)

### 2、Preliminaries

### 3、CQL

给出了数学基础：

![image-20250624162602016](img/image-20250624162602016.png)

### 4、Practical Algorithm and Implementation Details

![image-20250624162827873](img/image-20250624162827873.png)

### 5、Related Work

### 6、Experiments

![image-20250624163737837](img/image-20250624163737837.png)

论文中还有很多实验数据的展示，不一一誊录了。

### 7、Discussion

### 8、Bison的实验
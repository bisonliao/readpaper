**Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction**

### 1、Introduction

与 RL 不同的是，模仿学习（IL）把序列预测/决策问题简化为有监督学习：在训练阶段，通过我们会有一个最优的专家，它可以是一个策略函数Pi或者价值函数Q，任何时候它都能给出最优的动作。但这个专家只是在训练阶段存在，推理/测试阶段就要靠我们习得的agent自身的能力。

借助专家，IL通常比RL学的快很多。



模仿学习中的核心问题是：

- 一开始 agent 是跟专家学的，数据分布主要集中在 **专家访问过的状态**。
- 但当 agent 自己开始执行时，容易跑到训练时没见过的新状态，导致数据分布偏移（distribution mismatch）。

解决方案是“ interleave”，也就是：

- 训练的时候，也让 agent 自己测试，暴露它自己的错误状态，然后收集这些新状态的数据，继续学习。
- 相当于主动修正它自己会走偏的状态。

这正是 DAgger、AggreVaTe 这些算法的本质。一句话：一些交互式的模仿学习方法，例如 SEARN、DAgger 和 AggreVaTe，通过**交错地进行学习和测试的过程**，来解决训练和测试时状态分布不一致的问题。因此，这类方法在实际应用中通常效果更好。



本论文提出了AggreVaTeD方法，是AggreVaTe方法的改进版（而AggreVaTe又是DAgger的改进版）。实验证明AggreVaTeD方法可以获得专家级的效果，甚至超过专家，这是那些非交互式的模仿学习方法通常无法达到的性能。

这三者的一脉相承关系是这样的：

1. DAgger：数据集聚合，学专家的动作
2. 从纯行为模仿（DAgger）进化到 目标优化：学会最小化 future cost（更 RL 化的目标），就得到了AggreVaTe
3. 解决 AggreVaTe 可微和深度网络兼容问题，就得到了AggreVaTeD

这个鬼怎么读嘛？

![image-20250625113751254](img/image-20250625113751254.png)

### 2、Preliminaries

介绍了MDP和DRL的一些基础知识

### 3、Differentiable Imitation Learning

感觉这里的数学推导也没有什么新东西，有点故弄玄虚...

![image-20250625160257262](img/image-20250625160257262.png)



### 4、Sample-Based Practical Algorithms

#### 算法过程

![image-20250625151512758](C:\GitHub\readpaper\img\image-20250625151512758.png)

#### 与DAgger的对比

![image-20250625151745424](img/image-20250625151745424.png)

### 5、Quantify the Gap: An Analysis of IL vs RL

从理论上量化模仿学习（Imitation Learning, IL）相较于强化学习（Reinforcement Learning, RL）的学习效率优势

### 6、Expertiments

![image-20250625160635317](img/image-20250625160635317.png)
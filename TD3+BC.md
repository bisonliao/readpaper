**A Minimalist Approach to Offline Reinforcement Learning**

离线强化学习的极简实现

### 1、Introduction

离线强化学习（Offline Reinforcement Learning）指的是在一组固定的离线数据上训练智能体。由于对数据分布之外的动作（out-of-distribution actions）进行价值估计时容易产生误差，因此大多数离线强化学习算法都会通过约束或正则化策略网络，使其生成的动作尽可能接近离线数据中的动作分布。

将已有的在线强化学习算法改造为离线版本，通常会引入额外的复杂性。离线强化学习方法往往需要新增超参数、辅助模块（如生成模型），并对基础的在线RL算法结构进行调整。

本文的目标是在**尽量少改动的前提下**，让一个深度强化学习算法能够在离线环境中运行良好。我们发现，只需在策略更新中加入行为克隆项（behavior cloning term），并对数据进行标准化处理，即可将一个在线RL算法转化为一个离线RL算法，并达到与当前最先进算法相当的性能。该方法不仅易于实现和调试，而且由于省去了额外计算，大大降低了运行成本（节省了一半以上的训练时间）。



在线RL，需要agent与环境交互，这通常是昂贵的，而且未经训练的agent与环境交互是很有风险的（例如自动驾驶、自动手术）。离线RL使用历史上的log数据、或者利用人类专家的操作演示，避免了昂贵且高风险的环境交互。

离线RL算法的训练结果通常表现更差，由于对训练数据集以外的(s,a)的价值估计误差，agent会的这些分布以外的动作价值出现高估而更倾向于这些动作，导致性能表现差。

这类问题的解决方法通常是让 被训练的策略模型 尽量 接近行为模型（也就是产生训练数据的与环境交互的模型），方法通常叫做：batch限制、KL控制、行为正则、策略限制。

各种离线RL算法都很复杂难以实现。bison：类比一下RND、ICM，在基本RL算法基础上加了两三个神经网络，我就怎么都复现不出来好的实验结果。

本论文提出的算法，是在TD3算法的基础上做很小的改动（ use TD3 with behavior cloning (BC) for the purpose of offline RL）：

![image-20250617142742454](img/image-20250617142742454.png)

### 2、Related Work

我们是首先提出对TD3算法应用行为克隆而改造为offline算法的。但行业内对于把RL与BC或者其他模仿学习手段的研究也很普遍。

然后论文介绍了 RL+BC、RL+Imitation、Offline RL等等行业内进行的研究。

![image-20250617143654368](img/image-20250617143654368.png)

### 3、Background

介绍了RL、offline RL、BC:

BC. Another approach for training policies is through imitation of an expert or behavior policy. **Behavior cloning (BC)** is an approach for imitation learning , where the policy is trained with supervised learning to directly imitate the actions of a provided dataset. Unlike RL, this
process is highly dependent on the performance of the data-collecting process.



Most offline RL algorithms are built explicitly on top of an existing off-policy deep RL algorithm, such as TD3 or SAC , but then further modify the underlying algorithm.

### 4、Challenges in Offline RL

RL algorithms are notoriously difficult to implement and tune, where minor code-level optimizations and hyperparameters can have non-trivial impact of performance and stability. This problem may be additionally amplified in the context of offline RL

这章节从实现的复杂度、性能高低、额外的计算开销、训练的稳定性等维度，对三个离线RL算法做了分析比较。

### 5、A Minimalist Offline RL Algorithm

introduction部分有说，这里不重复。论文也没有更多算法描述

### 6、Experiments

#### 什么是D4RL数据集

D4RL（Datasets for Deep Data-Driven Reinforcement Learning）提供了多个经典强化学习任务的**离线经验数据集**

![image-20250617153112821](img/image-20250617153112821.png)

#### 实验对比效果

![image-20250617153549387](img/image-20250617153549387.png)

### 7、Conclusion

我自然要问，其他离策略算法，可以用论文中的类似思路改造为offline RL吗？

AI：

![image-20250617154201104](img/image-20250617154201104.png)
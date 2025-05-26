**Reincarnating Reinforcement Learning:Reusing Prior Computation to Accelerate Progress**

### Introduction

1. 在大型任务场景下，从头训练agent是一个计算密集、耗时长久的事情。而且通常随着模型和参数的调整导致反复从头训练，费时费力。例如训练一个agent玩转50款atari游戏，需要1000GPU天的训练；训练一个打星际争霸的agent，需要数百万美金的开销。
2. 再生RL（Reincarnating RL ）的思路是尽量利用已经训练的网络参数或者已经收集的环境交互数据，来加速训练的一种工作流模式。
3. 根据前期不同的数据形态（log下来的数据、已经训练的策略...），RRL有不同的方法。论文的工作关注的是从已经训练的策略网络迁移到价值网络的RRL方法（PVRL) 。
4. 为了从已有的策略网络“断奶”（断开依赖），我们使用QDagger方法，它结合了Dagger和n-steps Q-learning方法。



### Preliminaries

介绍了RL的基本概念，包括策略、价值函数、贝尔曼方程等等
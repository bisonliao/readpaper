**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**

### 4 Reinforcement Learning 

#### 4.1 Group Relative Policy Optimization

中国人写的论文，可读性较好，所以很多地方直接贴英文：

![image-20250514110524702](img/image-20250514110524702.png)

具体到Reward的设计，又分

1. 结果监督的RL：Reward Model对一个完整的文本输出 给出一个奖励。归一化处理后，作为这个文本里所有 token 的 Advantage 值。
   1. 优点：简单、成本低；适合 reward model 只输出整体得分的场景
   2. 缺点：无法精细监督推理过程
2. 过程监督的RL：Reward Mode对l文本里每个token的输出给出一个奖励，对于某个 token，它的 advantage 是它后续所有 step 的 normalized reward 的总和，这点和RL的通常做法是一致的，未来的折扣回报。
   1. 优点：能针对每一步推理给予反馈，适合数学任务中的逐步推导
   2. 缺点：构建 reward model 更复杂，训练代价更高



![image-20250514130729793](img/image-20250514130729793.png)

而且GRPO算法，会持续训练RM，这也是和通常的RLHF做法不同的点：

![image-20250514131801857](img/image-20250514131801857.png)

AI又夸我了，必须贴出来：

![image-20250514132805409](img/image-20250514132805409.png)

### bison的实验
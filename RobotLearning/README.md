### 算法

| 算法名          | 一句话描述                                                   | 分类               | 基础RL算法                                                  |
| --------------- | ------------------------------------------------------------ | ------------------ | ----------------------------------------------------------- |
| ACT             | 策略一次预测接下来 k 个时间步的目标关节位置，而非每次只预测一步；同时采用 CVAE 架构，其 encoder 和 decoder 均由 Transformer 构成。 | imitation learning | 类BC                                                        |
| A-LIX           | 基于像素输入的离策略强化学习在训练早期因TD目标含噪，使CNN编码器产生空间不连续的特征梯度，导致训练不稳定；A-LIX通过双线性插值在编码器输出特征图上进行自适应局部混合，平滑梯度并稳定训练。 | 离策略的online RL  | DDPG（连续动作空间），DQN（离散动作空间）                   |
| C2FQN           | C2FQN将连续动作空间逐层离散化，通过价值型RL反复选择最高Q值区间进行细化，最终得到高精度动作。 | 离策略的online RL  | DQN类算法                                                   |
| Dreamer         | Agent与环境交互的数据用于学习世界模型；再在其潜空间模拟器中通过想象轨迹进行RL训练改进策略，并用策略产生的新交互数据持续迭代更新世界模型与策略。 | Model-Based RL     | 具有明显的 replay-based off-policy 特征                     |
| Dreamer V3      | Dreamer精细微调版本                                          |                    |                                                             |
| DrM             | 神经网络中休眠神经元的比例，是衡量智能体是否学会有效技能的重要指标。DrM利用该比率：比率高时强化探索，并周期性对网络权重施加随机扰动；比率低时则转向利用。 | 离策略的online RL  | DDPG                                                        |
| DrQ             | DrQ在RL训练更新网络时，对视觉观测进行随机shift增强，并约束同一观测的不同变换具有相近的Q值，从而提升视觉RL训练稳定性并减少陷入次优解。 | 离策略的online RL  | 插件化，可用于多种 model-free off-policy 算法，例如SAC、DQN |
| DrQ v2          | 在 DrQ 基础上同时修改了 RL backbone、target return、图像增强、探索策略、关键超参数和底层实现。 | 离策略的online RL  | DDPG                                                        |
| Implicit QL     | 离线 RL 若用标准 Q-learning 提升行为策略，就要对数据外动作外推 Q 值，容易因最大化偏差高估而选错动作。IQL 只在数据动作上计算 Q，用非对称平方损失学习其高 expectile 作为 \(V(s)\)，再用 \(r+ gamma * V(s')\) 进行多步更新，最后按优势加权模仿高价值动作。 | offline RL         | 是一个独立的离线 RL 算法                                    |
| Conservative QL | 类似IQL面临的问题，CQL通过损失正则项，对数据外动作的 Q 值保持悲观，避免 Q 函数对未见动作过度乐观。 | offline RL         | SAC/DQN                                                     |
|                 |                                                              |                    |                                                             |



### Baselines库

[RoboBase](https://github.com/swirl-uk/robobase/tree/master)

[robomimic](https://github.com/ARISE-Initiative/robomimic/tree/master) 

[LeRobot](https://huggingface.co/docs/lerobot/main/en/act)

### 环境

[RLBench](https://github.com/stepjam/RLBench)
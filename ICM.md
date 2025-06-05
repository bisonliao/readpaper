**Curiosity-driven Exploration by Self-supervised Prediction**

### Introduction

还是怎么解决奖励稀疏问题的老生常谈。

本论文提出一种类似RND的使用额外的深度神经网络 产生 内部奖励的方法，命名为 Intrinsic Curiosity Module (ICM)。

相比类似的方法，它有额外的优势：

1. As there is no incentive for this feature space to encode any environmental features that are not influenced by the agent’s actions, our agent will receive no rewards for reaching environmental states that are inherently unpredictable and its exploration strategy will be robust to the presence of distractor objects, changes in illumination, or other nuisance sources of variation in the environment.
2. Even in the absence of any extrinsic rewards, a curious agent learns good exploration policies.
3. the proposed method enables an agent to learn generalizable skills even in the absence of an explicit goal: the exploration policy learned in the first level of Mario helps the agent explore subsequent levels faster.

### Curiosity-Driven Exploration

#### 算法原理

终于遇到一篇论文践行了“一图胜千言”的方法论：

![image-20250605144404832](img/image-20250605144404832.png)

#### 直观理解

RND作为一种内部激励的产生方式，从直觉上好理解：目标神经网络和预测神经网络的输出的差作为内部激励，由于目标网络是冻结参数的，而预测网络是会以目标网络的输出作为label来学习的，所以对于出现过的state，他们的预测会很接近，对于没有出现过的state，他们的预测会有差距。所以差距的大小会体现state的新奇程度。



ICM通过**预测环境动态的变化**来生成内在奖励，其核心是：

- **只关注与Agent动作相关的环境变化**，忽略无关因素（如风吹树叶、光照变化等）。
- 通过两个子模型实现：
  - **逆动力学模型（Inverse Model）**：学习从状态变化中反推动作（即“当前状态 + 下一状态 → 动作”）。
  - **前向动力学模型（Forward Model）**：预测下一状态的特征（即“当前状态 + 动作 → 预测下一状态”）。

**内在奖励**定义为前向模型的预测误差（预测状态与实际状态的差异）。误差越大，说明当前状态-动作对越“新奇”。



两者的直觉差异：

- **RND的直觉**：“没见过的东西就是新奇的。”
- **ICM的直觉**：“我无法预测我的动作会带来什么后果，所以我要探索。”
  → 更注重​**​动作的因果效应​**​，而非单纯的状态新奇性。



ICM适合需要**学习环境动态**的任务（如导航、交互），而RND更适合**覆盖未知状态空间**的任务（如开放世界探索）。两者本质都是“用预测误差驱动探索”，但ICM通过逆模型实现了更精准的因果关联。

### Expertimental Setup

#### 环境

用的是VizDoom，也是一个DRL模拟环境。[详细文档在这里](https://vizdoom.farama.org/)

API大概是这样的风格：

```python
import vizdoom as vzd
game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"))
game.init()
for _ in range(1000):
   state = game.get_state()
   action = policy(state)  # this is where you would insert your policy
   reward = game.make_action(action)

   if game.is_episode_finished():
      game.new_episode()

game.close()
```

#### 网络结构

超参数：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 全局超参数（来自论文Section 3）
class Config:
    # 输入预处理
    INPUT_SIZE = (42, 42)          # 输入图像resize尺寸
    INPUT_CHANNELS = 4             # 堆叠的帧数（当前帧 + 过去3帧）
    ACTION_REPEAT = 4              # VizDoom的动作重复次数
    GRAYSCALE = True               # 是否转为灰度图

    # A3C网络参数
    A3C_CONV_FILTERS = [32, 32, 32, 32]  # 4层卷积的滤波器数
    A3C_CONV_KERNEL = 3           # 卷积核大小
    A3C_CONV_STRIDE = 2           # 卷积步长
    A3C_CONV_PADDING = 1          # 卷积padding
    A3C_LSTM_UNITS = 256           # LSTM隐藏单元数

    # ICM网络参数
    ICM_FEATURE_DIM = 288          # 逆模型输出的特征维度
    ICM_FORWARD_FC_UNITS = 256    # 前向模型的FC层单元数
    ICM_INVERSE_FC_UNITS = 256    # 逆模型的FC层单元数
    ICM_BETA = 0.2                 # 前向/逆模型损失权重（Equation 7）
    ICM_LAMBDA = 0.1               # 策略梯度与ICM损失的权重（Equation 7）
    ICM_ETA = 1.0                  # 内在奖励缩放因子（Equation 6）

    # 训练参数
    LR = 1e-3                      # 学习率（ADAM优化器）
    GAMMA = 0.99                   # 奖励折扣因子
    ENTROPY_BETA = 0.01           # 熵正则化系数
```

RL部分是带LSTM的A3C架构

```python
class A3CNetwork(nn.Module):
    def __init__(self, num_actions):
        super(A3CNetwork, self).__init__()
        # 卷积层（4层，参数来自论文）
        self.conv = nn.Sequential(
            nn.Conv2d(Config.INPUT_CHANNELS, Config.A3C_CONV_FILTERS[0], 
                      kernel_size=Config.A3C_CONV_KERNEL, 
                      stride=Config.A3C_CONV_STRIDE, 
                      padding=Config.A3C_CONV_PADDING),
            nn.ELU(),
            nn.Conv2d(Config.A3C_CONV_FILTERS[0], Config.A3C_CONV_FILTERS[1], 
                      kernel_size=Config.A3C_CONV_KERNEL, 
                      stride=Config.A3C_CONV_STRIDE, 
                      padding=Config.A3C_CONV_PADDING),
            nn.ELU(),
            nn.Conv2d(Config.A3C_CONV_FILTERS[1], Config.A3C_CONV_FILTERS[2], 
                      kernel_size=Config.A3C_CONV_KERNEL, 
                      stride=Config.A3C_CONV_STRIDE, 
                      padding=Config.A3C_CONV_PADDING),
            nn.ELU(),
            nn.Conv2d(Config.A3C_CONV_FILTERS[2], Config.A3C_CONV_FILTERS[3], 
                      kernel_size=Config.A3C_CONV_KERNEL, 
                      stride=Config.A3C_CONV_STRIDE, 
                      padding=Config.A3C_CONV_PADDING),
            nn.ELU()
        )

        # LSTM层（256单元）
        self.lstm = nn.LSTMCell(Config.A3C_CONV_FILTERS[-1] * 6 * 6, Config.A3C_LSTM_UNITS)  # 6x6是卷积后的空间尺寸

        # 策略头（Actor）
        self.actor = nn.Linear(Config.A3C_LSTM_UNITS, num_actions)

        # 值函数头（Critic）
        self.critic = nn.Linear(Config.A3C_LSTM_UNITS, 1)

    def forward(self, x, hx, cx):
        # 输入x: [batch_size, 4, 42, 42]
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        hx, cx = self.lstm(x, (hx, cx))
        action_probs = torch.softmax(self.actor(hx), dim=-1)
        value = self.critic(hx)
        return action_probs, value, hx, cx
```

ICM部分:

```python
class ICM(nn.Module):
    def __init__(self, num_actions):
        super(ICM, self).__init__()
        # 共享的特征编码器（4层卷积，与A3C相同但独立）
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(Config.INPUT_CHANNELS, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU()
        )

        # 逆模型（动作预测）
        self.inverse_model = nn.Sequential(
            nn.Linear(Config.ICM_FEATURE_DIM * 2, Config.ICM_INVERSE_FC_UNITS),
            nn.ELU(),
            nn.Linear(Config.ICM_INVERSE_FC_UNITS, num_actions)
        )

        # 前向模型（状态预测）
        self.forward_model = nn.Sequential(
            nn.Linear(Config.ICM_FEATURE_DIM + num_actions, Config.ICM_FORWARD_FC_UNITS),
            nn.ELU(),
            nn.Linear(Config.ICM_FORWARD_FC_UNITS, Config.ICM_FEATURE_DIM)
        )

    def forward(self, state, next_state, action):
        # 编码状态特征
        phi_state = self.feature_encoder(state)
        phi_state = phi_state.view(phi_state.size(0), -1)  # [batch, 288]
        phi_next_state = self.feature_encoder(next_state)
        phi_next_state = phi_next_state.view(phi_next_state.size(0), -1)

        # 逆模型损失
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        pred_action = self.inverse_model(inverse_input)
        inverse_loss = nn.CrossEntropyLoss()(pred_action, action)

        # 前向模型损失
        forward_input = torch.cat([phi_state, action], dim=1)
        pred_phi_next_state = self.forward_model(forward_input)
        forward_loss = 0.5 * (pred_phi_next_state - phi_next_state.detach()).pow(2).mean()

        # 内在奖励（特征空间的预测误差）
        intrinsic_reward = Config.ICM_ETA * 0.5 * (pred_phi_next_state - phi_next_state).pow(2).mean(dim=1)

        return intrinsic_reward, inverse_loss, forward_loss
```

优化器：

```python
def build_optimizers(a3c_net, icm_net):
    # A3C优化器（策略梯度 + 值函数损失）
    a3c_optimizer = optim.Adam(a3c_net.parameters(), lr=Config.LR)

    # ICM优化器（前向 + 逆模型损失）
    icm_optimizer = optim.Adam(icm_net.parameters(), lr=Config.LR)

    return a3c_optimizer, icm_optimizer
```

#### ELU激活函数

我是第一次接触到ELU激活函数：

![image-20250605155220463](img/image-20250605155220463.png)

#### LSTM的使用

![image-20250605161017731](img/image-20250605161017731.png)

#### Baseline方法

实验对比三者：

1. ICM + A3C：完整的ICM实现和A3C
2. A3C：普通的A3C，不带内部激励机制
3. ICM-pixels + A3C：ICM只使用forward网络，不使用逆向网络

### Experiments

#### 不同奖励密度程度下的表现

![image-20250605162046937](img/image-20250605162046937.png)

#### 对干扰因素的鲁棒性

![image-20250605162434557](img/image-20250605162434557.png)

#### 无奖励场景

A good exploration policy is one which allows the agent to visit as many states as possible even without any goals.In
the case of 3-D navigation, we expect a good exploration policy to cover as much of the map as possible; in the case of playing a game, we expect it to visit as many game states as possible. 

![image-20250605165908011](img/image-20250605165908011.png)

论文还提到：超级马里奥游戏里，完全没有外部奖励的情况下，agent只用ICM学会了杀敌，因为不杀敌会死，从而不能看到很多状态。

#### 泛化能力

### Related Work

列举了业界各种Curiosity-Driven Exploration的工作

### Conclusion

未来研究的一个有趣方向是将学习到的探索行为/技能作为更复杂、分层系统中的运动原始/低级策略。例如，我们的VizDoom代理学会沿着走廊行走，而不是撞到墙上。这可能是导航系统的有用基础。

### bison的实验

#### 疯狂的赛车
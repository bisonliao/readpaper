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

也是Farama Foundation这个非盈利组织提供的。

![image-20250617153112821](img/image-20250617153112821.png)

#### 实验对比效果

![image-20250617153549387](img/image-20250617153549387.png)

### 7、Conclusion

我自然要问，其他离策略算法，可以用论文中的类似思路改造为offline RL吗？

AI：

![image-20250617154201104](img/image-20250617154201104.png)

### 8、bison的实验

#### pendulum

step1：先用SAC算法训练一个expert，然后让expert在eval模式下与环境交互收集经验数据，保存下来。这里不贴详细代码了，用之前的SAC的代码即可快速搞定。

step2：使用TD3+BC算法，进行offline RL。

我靠，学习效果很好，一把成功，下面是评估函数的输出，玩过倒立摆的人都知道，一个回合-200左右的回报是训练有素的agent了：

```
Evaluation Episode 1, Reward: -0.15
Evaluation Episode 2, Reward: -123.72
Evaluation Episode 3, Reward: -236.22
Evaluation Episode 4, Reward: -231.89
Evaluation Episode 5, Reward: -240.85
```

![image-20250617191731549](img/image-20250617191731549.png)

TD3+BC的代码如下（在TD3的基础上，改动真的好小，而且不复杂）：

```python
import datetime
import os
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子保证可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# 超参数配置
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENV_NAME = "Pendulum-v1"
    MAX_EPISODES = 10000  # 最大训练回合数
    MAX_STEPS = 200  # 每回合最大步数
    BATCH_SIZE = 256  # 从经验池采样的批次大小
    GAMMA = 0.99  # 折扣因子
    TAU = 0.005  # 目标网络软更新系数
    LR_ACTOR = 1e-4  # Actor学习率
    LR_CRITIC = 1e-3  # Critic学习率
    REPLAY_BUFFER_SIZE = 100000  # 经验回放池大小
    EXPLORATION_NOISE = 0.1  # 探索噪声标准差
    POLICY_NOISE = 0.1  # 目标策略平滑噪声标准差
    NOISE_CLIP = 0.3  # 噪声截断范围
    POLICY_UPDATE_FREQ = 2  # 策略网络延迟更新频率
    CHECKPOINT_DIR = "./td3_checkpoints"  # 模型保存目录
    CHECKPOINT_INTERVAL = 50  # 每隔多少回合保存一次模型
    LOGS_DIR = "./logs"


# 创建检查点目录
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.LOGS_DIR, exist_ok=True)
writer = SummaryWriter(f"{Config.LOGS_DIR}/td3_bc_pendulum_{datetime.datetime.now().strftime('%m%d_%H%M%D')}")


# Actor策略网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        # 网络结构
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        """
        输入: state [batch_size, state_dim]
        输出: action [batch_size, action_dim], 范围[-max_action, max_action]
        """
        state = state / torch.tensor([[1.0, 1.0, 8.0]]).to(Config.DEVICE)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a


# Critic价值网络 (双Q网络)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1网络
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2网络 (结构相同但参数独立)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        输入:
            state [batch_size, state_dim]
            action [batch_size, action_dim]
        输出:
            Q1值 [batch_size, 1], Q2值 [batch_size, 1]
        """

        sa = torch.cat([state, action], dim=1)
        sa = sa / torch.tensor([[1.0, 1.0, 8.0, 2.0]]).to(Config.DEVICE)

        # Q1网络前向
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2网络前向
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        """仅返回Q1值"""
        sa = torch.cat([state, action], dim=1)
        sa = sa / torch.tensor([[1.0, 1.0, 8.0, 2.0]]).to(Config.DEVICE)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# 经验回放池
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """存储单步经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样批次经验"""
        batch = random.sample(self.buffer, batch_size)
        # 转换为PyTorch张量并移动到设备
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(Config.DEVICE)
        actions = torch.FloatTensor(np.array([t[1] for t in batch])).to(Config.DEVICE)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).unsqueeze(1).to(Config.DEVICE)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(Config.DEVICE)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).unsqueeze(1).to(Config.DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# TD3算法主体
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action

        # 初始化网络
        self.actor = Actor(state_dim, action_dim, max_action).to(Config.DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(Config.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=Config.LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(Config.DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(Config.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=Config.LR_CRITIC)

        # 经验回放池
        self.replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)

        # 训练计数器
        self.total_it = 0
        self.lambda_ = 1.0

    def select_action(self, state, add_noise=True):
        """
        根据状态选择动作 (推理时add_noise=False)
        输入: state [state_dim] (numpy数组)
        输出: action [action_dim] (numpy数组)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)  # [1, state_dim]
        action = self.actor(state).cpu().data.numpy().flatten()  # [action_dim]

        if add_noise:
            # 添加探索噪声
            noise = np.random.normal(0, Config.EXPLORATION_NOISE, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def train(self, batch_size):
        """训练一步"""
        self.total_it += 1

        # 从经验池采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # 计算目标Q值 (带策略平滑和双Q网络最小值)
        with torch.no_grad():
            # 目标策略动作 + 平滑噪声
            noise = (torch.randn_like(actions) * Config.POLICY_NOISE).clamp(
                -Config.NOISE_CLIP, Config.NOISE_CLIP)
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action)

            # 双Q网络目标值取最小
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * Config.GAMMA * target_Q

        # 更新Critic网络 (最小化TD误差)
        current_Q1, current_Q2 = self.critic(states, actions) #type:torch.Tensor
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        writer.add_scalar("train/current_Q1", current_Q1.mean().cpu().item(), self.total_it)
        writer.add_scalar("train/current_Q2", current_Q2.mean().cpu().item(), self.total_it)

        self.lambda_ = 2.5 / current_Q1.abs().mean().item()
        writer.add_scalar('train/lambda', self.lambda_, self.total_it)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) #对梯度进行裁剪，防止梯度爆炸
        self.critic_optimizer.step()
        writer.add_scalar("train/critic_loss", critic_loss.cpu().item(), self.total_it)

        # 延迟更新Actor和目标网络
        if self.total_it % Config.POLICY_UPDATE_FREQ == 0:
            # 更新Actor (最大化Q值)
            new_actions = self.actor(states)
            bc_regulation = F.mse_loss(new_actions, actions)
            actor_loss = -self.critic.Q1(states, new_actions).mean() * self.lambda_ + bc_regulation

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0) #对梯度进行裁剪，防止梯度爆炸
            self.actor_optimizer.step()

            writer.add_scalar("train/actor_loss", actor_loss.cpu().item(), self.total_it)

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(Config.TAU * param.data + (1 - Config.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(Config.TAU * param.data + (1 - Config.TAU) * target_param.data)

# 训练函数
def train_td3(resume_checkpoint=None):
    # 初始化环境
    env = gym.make(Config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()
    del env

    # 初始化TD3
    td3 = TD3(state_dim, action_dim, max_action)

    start_episode = 0

    # 加载专家离线数据集
    experience = torch.load('./pendulum_expert_experience.pth', weights_only=False)
    for e in experience:
        state, action, reward, next_state, terminated, truncated, info = e
        done = terminated or truncated
        td3.replay_buffer.add(state, action, reward, next_state, done)

    for episode in range(start_episode, Config.MAX_EPISODES):
            td3.train(Config.BATCH_SIZE)

    eval_td3(td3)


# 推理函数 (加载模型并渲染)
def eval_td3(td3:TD3,  num_episodes=5):
    # 初始化环境 (带渲染)
    env = gym.make(Config.ENV_NAME, render_mode="human")

    td3.actor.eval()
    # 推理循环
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(Config.MAX_STEPS):
            action = td3.select_action(state, add_noise=False)  # 推理时不加噪声
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward:.2f}")
    td3.actor.train()
    env.close()


# 主函数
def main(arg):
    train_td3()
    
main("train")
```


**Soft Actor-Critic:Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**

model-free的深度强化学习算法已经在很多有挑战的决策与控制任务中展现，但这些方法遇到了两个很典型的挑战：

1. 非常高的抽样（样本）复杂性。智能体需要与环境进行极大量交互（通常数百万至数十亿次）才能学到有效策略
2. 脆弱的收敛，对细致的调参有很高的要求

这两个挑战严重的影响了DRL方法在复杂真实场景的落地。

这篇论文，我们提出了soft actor-critic方法：一种off-policy的基于actor-critic方法的熵最大化的深度强化学习方法。actor除了最大化reward期望值，还要最大化熵：让动作尽量随机。SAC通过离线策略数据重用和随机策略的熵正则化，解决了连续控制任务中样本效率与稳定性的平衡问题我们的方法在很多连续的benchmark任务里取得了先进的表现。而且，相比之前其他off-policy方法，我们的算法非常稳定，使用很多不同的随机种子做实验都获得了相同的先进表现。

bison:   在深度强化学习（Deep Reinforcement Learning, DRL）中，**Model-free** 意味着智能体（agent）对马尔可夫决策过程（MDP）的**转移概率矩阵 P(s′∣s,a)** 和**奖励函数 R(s,a,s′)** 完全未知，只能通过与环境的交互来学习策略或价值函数。与之对应的是 **Mode-Based**

### introduction

如前面所说的DRL普遍应用和两个挑战。

on-policy learning是导致非常高的抽样复杂性的一个原因，每次policy梯度下降，都要求收集新的样本，智能体重新与环境交互。off-policy learning可以反复利用抽样的样本，但off-policy方法对于传统的策略梯度类算法并不直接可行，对于Q Learning类算法是直接可用的。

off-policy learning与高维度的非线性函数拟合器（例如深度神经网络就是一种）结合，会面临不稳定和难以收敛的挑战，尤其是在连续的state空间和连续的动作空间。这种场景下，常用的DDPG算法提供样本高效率的同时，对超参数非常敏感和脆弱。

最大熵RL方法修改了RL的目标函数，引入最大熵使得探索和鲁棒性有非常显著的提升。行业内之前的工作显示：最大熵的做法在off-policy和on-policy方法里都适用。

bison：原文绕来绕去的车轱辘话，一言以蔽之：off-policy解决样本效率问题，最大熵解决off-policy随之而来的稳定性问题。

SAC方法可以轻松的扩展到一些高维的非常复杂的任务场景中，例如Humanoid benchmark，它的动作空间有21维，DDPG算法在这种测试下难有好的结果。

SAC方法有三个关键点：

1. 连续动作空间的actor-critic架构。actor网络输出的是动作的概率分布（高斯分布的均值与标准差）而不是确定性的动作
2. off-policy方法，提高样本效率
3. 最大化熵，提升稳定度、加强探索

### From Soft Policy Iteration to Soft Actor-Critic

这一节看不懂，似乎又是论文的核心，所以让AI帮忙解读：

![image-20250413121615283](img/image-20250413121615283.png)

#### 算法

论文中算法描述太简单，我请AI帮我给出了详细的SAC算法版本：

![](img/image-20250409114108202.png)

### Experiments

#### 对比评估

SAC与其他几个SOTA算法的实验对比：

1. deep deterministic policy gradient (DDPG) 
2. proximal policy optimization (PPO) 
3. soft Q-learning(SQL) 
4. twin delayed deep deterministic policy gradient algorithm (TD3)

![image-20250413122347040](img/image-20250413122347040.png)

#### 措施消减评估

下图表明：随机性的策略输出可以提升训练的稳定性。确定性的SAC版本（红色）显示不同初始化随机种子下的平均回报差异巨大。

![image-20250413124942505](img/image-20250413124942505.png)

下面的图表表明：

1. 推理的时候，关闭了熵正则，直接使用概率密度最高的均值作为动作，相比训练时候引入随机熵，推理获得更高的回报
2. SAC对reward值缩放敏感，需要为每个任务单独调试缩放倍数。
3. SAC对目标网络软更新系数，太大了会不稳定，太小了训练效率慢

![image-20250413125647366](img/image-20250413125647366.png)

### Conclusion

我们提出了soft actor-critic算法，这是一种基于最大熵框架的off-policy深度强化学习方法，能够在保持熵最大化的优势的同时，实现高效的样本利用率和训练稳定性。我们的理论分析证明了软策略迭代（Soft Policy Iteration）的收敛性，并基于此理论推导出实用的SAC算法。实验结果表明，SAC在性能上显著优于当前最先进的离线策略（如DDPG）和在线策略（如PPO）深度强化学习方法，并且在样本效率方面远超DDPG。

我们的研究结果表明，随机策略+熵最大化的强化学习方法能够显著提升算法的鲁棒性和稳定性。未来，进一步探索最大熵方法（如结合二阶优化信息或更复杂的策略表示）将是一个极具潜力的研究方向。

### bison的实验

##### BipedalWalker

训练了约24万步，表现还可以

![image-20250604161325063](img/image-20250604161325063.png)

showcase的日志：

```
Training completed. evaluating...
total reward:306.76385498046875
total reward:307.753173828125
total reward:309.64227294921875
total reward:308.17364501953125
total reward:3.774566650390625
```

代码如下：

```python

import gymnasium as gym
from gymnasium import Env
import gymnasium_robotics
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

from  stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter



class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SAC:
    def __init__(self, state_dim, action_dim, env:Env, device="cpu", writer=None):
        # set hyperparameters
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.writer = writer  # type:SummaryWriter
        self.update_count = 0
        self.action_dim = action_dim

        # initialize actor
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)

        # initialize critics - we will use double Q network clipping trick
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)

        # copy weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # adam optimizers lr=3e-4
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=1e-3)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=1e-3)

        # entropy target
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

        self.replay_buffer = ReplayBuffer(1_000_000, env.observation_space, env.action_space, device, handle_timeout_termination=False)

    def select_action(self, state):

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy.sample(state_tensor)
            action = action.cpu().numpy()[0]
            action = np.clip(action, -1, 1)
        return action

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return

        self.update_count += 1

        state_batch = batch.observations
        action_batch = batch.actions
        next_state_batch = batch.next_observations
        reward_batch = batch.rewards
        done_batch = batch.dones

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Update Q networks
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_state_batch)
            next_q1 = self.q1_target(next_state_batch, next_action)
            next_q2 = self.q2_target(next_state_batch, next_action)
            # Q network clipping trick
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q

        # Q1 update
        current_q1 = self.q1(state_batch, action_batch)
        q1_loss = F.mse_loss(current_q1, target_q)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Q2 update
        current_q2 = self.q2(state_batch, action_batch)
        q2_loss = F.mse_loss(current_q2, target_q)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy update
        new_action, log_prob = self.policy.sample(state_batch)
        q1_new = self.q1(state_batch, new_action)
        q2_new = self.q2(state_batch, new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_prob - q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        if self.writer and self.update_count % 200 == 0:
            self.writer.add_scalar('SAC/q1_loss', q1_loss.item(), self.update_count)
            self.writer.add_scalar('SAC/q2_loss', q2_loss.item(), self.update_count)
            self.writer.add_scalar('SAC/policy_loss', policy_loss.item(), self.update_count)
            self.writer.add_scalar('SAC/alpha_loss', alpha_loss.item(), self.update_count)
            self.writer.add_scalar('SAC/alpha', self.alpha.item(), self.update_count)

        # Update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def set_seed(seed):
    random.seed(seed)  # Python 的随机数种子
    np.random.seed(seed)  # NumPy 的随机数种子

    torch.manual_seed(seed)  # PyTorch 的 CPU 随机数种子
    torch.cuda.manual_seed(seed)  # PyTorch 的 GPU 随机数种子（当前 GPU）
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

    # 保证在 GPU 上的运算 deterministic（会牺牲一些性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def show(sac:SAC):

    env = gym.make("BipedalWalker-v3", render_mode='human')

    sac.policy.eval()

    for _ in range(5):# 多实验几次
        state, _= env.reset()
        total_reward = 0
        for i in range(2000):
            with torch.no_grad():
                action = sac.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            env.render()
            if terminated or truncated:
                break
            state = next_state
        print(f"total reward:{total_reward}")
    sac.policy.train()

def main():
    SEED = 43
    set_seed(SEED)
    # Set up environment
    env = gym.make('BipedalWalker-v3', render_mode=None)
    obs = env.reset(seed=SEED)[0]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"action_dim:{action_dim}, state_dim:{state_dim}")

    writer = SummaryWriter(log_dir=f'./logs/sac_bipedalwalker_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}')

    # Device setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Initialize SAC agent
    sac = SAC(state_dim, action_dim, env, device=device, writer=writer)


    # Set hyperparameters
    max_episodes = 200  # max number of episodes to stop training
    steps_cnt = 0
    episode_length = 2000
    batch_size = 256
    success_record = deque(maxlen=100) #保留过去100个回合的胜负

    for episode in range(max_episodes):

        obs, _ = env.reset()
        episode_reward = 0
        episode_len = 0
        success = 0
        for t in range(episode_length):
            # Create the state input by concatenating observation and desired goal
            state = obs
            # 跟踪输入的scale，如果太离谱，要做rescale
            if steps_cnt % 1000 == 3:
                writer.add_scalar("state/min", state.min(), steps_cnt)
                writer.add_scalar("state/max", state.max(), steps_cnt)
                writer.add_scalar("state/mean", state.mean(), steps_cnt)

            # Select action
            action = sac.select_action(state)

            # Step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps_cnt += 1
            episode_len += 1


            # Append transition to buffer
            sac.replay_buffer.add( obs, next_obs, action, np.array(reward), np.array(done), {})
            obs = next_obs
            episode_reward += reward
            if reward > 200: #不能判断等于300，因为每次移动施加扭矩，都会有一定的开销
                success = 1

            sac.update(batch_size)

            if done:
                break

        success_record.append(success)

        writer.add_scalar("episode/succeed_ratio", success_record.count(1) / 100.0, episode)
        writer.add_scalar("episode/episode_reward", episode_reward, episode)
        writer.add_scalar("episode/episode_len", episode_len, episode)
        if episode % 20 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    print("Training completed. evaluating...")
    env.close()
    show(sac)
    torch.save(sac, "./sac.pkl")


if __name__ == "__main__":
    main()
```


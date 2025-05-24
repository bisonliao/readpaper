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

下面的PPO和RPO都能够很好的收敛。由于pendulum任务比较简单，也没有体现RPO的明显优势：

![image-20250524171509348](img/image-20250524171509348.png)

代码如下：

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Normal, Independent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
import pygame

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=f"logs/Pendulum_PPO_{dt.now().strftime('%y%m%d_%H%M%S')}")

alpha = 0.3

def perturb_mean(mean):
    noise = torch.rand_like(mean)*2-1.0
    noise = (noise * alpha).to(device)
    return mean+noise



class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim // 2, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x / torch.tensor([1.0, 1.0, 8.0], device=x.device)  # normalize
        x = self.net(x)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    values = values + [0.0]  # bootstrap
    gae, returns = 0, []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        returns.insert(0, gae + values[t])
    return torch.tensor(returns, device=device)

def show(filename):

    env = gym.make("Pendulum-v1", render_mode='human')
    state_dim = env.observation_space.shape[0]
    policy = PolicyNetwork(state_dim).to(device)
    policy = torch.load(filename, weights_only=False)
    state, _ = env.reset()
    for i in range(2000):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, std = policy(state_tensor.unsqueeze(0))
        action_raw = mean
        action = torch.tanh(action_raw) * 2.0
        action_np = action.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        env.render()
        state = next_state


def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]

    policy = PolicyNetwork(state_dim).to(device)
    value_fn = ValueNetwork(state_dim).to(device)
    optim_policy = optim.Adam(policy.parameters(), lr=3e-4)
    optim_value = optim.Adam(value_fn.parameters(), lr=1e-3)

    step_count = 0

    while step_count < 600_000:
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        state, _ = env.reset()

        while len(states) < 2048:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                mean, std = policy(state_tensor.unsqueeze(0))
            mean = perturb_mean(mean)

            dist = Independent(Normal(mean, std), 1)
            action_raw = dist.rsample()
            action = torch.tanh(action_raw) * 2.0
            log_prob = dist.log_prob(action_raw)

            value = value_fn(state_tensor.unsqueeze(0)).item()

            action_np = action.cpu().numpy().astype(np.float32) # action.cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action_np[0])
            reward = torch.tensor(reward, dtype=torch.float32)
            done = terminated or truncated

            states.append(state_tensor)
            actions.append(action_raw.squeeze(0))
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.squeeze(0))
            values.append(value)

            state = next_state
            step_count += 1
            if done:
                state, _ = env.reset()

        with torch.no_grad():
            next_value = value_fn(torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)).item()
        values.append(next_value)
        returns = compute_gae(rewards, values, dones)
        values = torch.tensor(values[:-1], device=device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为张量
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs_old = torch.stack(log_probs)

        for _ in range(10):  # PPO更新迭代次数
            mean, std = policy(states)
            mean = perturb_mean(mean)
            dist = Independent(Normal(mean, std), 1)
            log_probs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy

            value_preds = value_fn(states)
            value_loss = nn.functional.mse_loss(value_preds, returns)

            optim_policy.zero_grad()
            policy_loss.backward()
            optim_policy.step()

            optim_value.zero_grad()
            value_loss.backward()
            optim_value.step()

        writer.add_scalar("loss/policy", policy_loss.item(), step_count)
        writer.add_scalar("loss/value", value_loss.item(), step_count)
        writer.add_scalar("stats/returns", sum(rewards), step_count)
        writer.flush()

        print(f"Step: {step_count}, Return: {sum(rewards):.2f}, Policy Loss: {policy_loss.item():.3f}")

    env.close()
    torch.save(policy, "./checkpoints/Pendulum_RPO.pth")



if __name__ == "__main__":
    train()
    show("./checkpoints/Pendulum_RPO.pth")

```


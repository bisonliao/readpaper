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

![image-20250625151512758](img/image-20250625151512758.png)

#### 与DAgger的对比

![image-20250625151745424](img/image-20250625151745424.png)

#### 实现细节

![image-20250626112507394](img/image-20250626112507394.png)

### 5、Quantify the Gap: An Analysis of IL vs RL

从理论上量化模仿学习（Imitation Learning, IL）相较于强化学习（Reinforcement Learning, RL）的学习效率优势

### 6、Expertiments

![image-20250625160635317](img/image-20250625160635317.png)

### 7、 Bison的实验

#### BipedalWalkerHardCore





代码如下：

```python
import datetime
import random

from stable_baselines3 import SAC

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F




writer = SummaryWriter(log_dir=f'logs/AggreVateD_BipedalWalker_{datetime.datetime.now().strftime("%m%d_%H%M%S")}')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Config:
    max_iteration = 300
    max_experience_len = 100_000
    state_dim = 24
    action_dim = 4
    max_action = 1
    K = 20
    H = 1000
    env_name = "BipedalWalkerHardcore-v3"
    hidden_dim = 128
    lr = 1e-4
    gamma = 0.99
    batch_sz = 128
    train_repeat = 50

# ----------------------------
# 1. 策略网络定义
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-10, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, Config.hidden_dim)
        self.fc2 = nn.Linear(Config.hidden_dim, Config.hidden_dim)

        self.mean = nn.Linear(Config.hidden_dim, action_dim)
        self.log_std = nn.Linear(Config.hidden_dim, action_dim)

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

    # 带随机性的采样动作和该动作的log-prob值
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        log_prob = normal.log_prob(x)
        # This is the crucial part, known as the "log-derivative of the tanh transformation" or
        # the "correction term for squashing functions".
        # It help to get the log_prob of the squashed action (action) in the squashed space
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # todo：如果任务的action幅度不是+-1，还需要乘，并调整log_prob
        return action, log_prob
    # 确定性的返回最大概率的动作
    def predict(self, state):
        mean, std = self.forward(state)
        x = mean
        action = torch.tanh(x)
        return action
    # 根据state计算动作分布，返回old_action动作在该分布下的log_prob，
    # old_action是经过tanh rescaled了的
    def get_log_prob(self, state, old_action):
        #避免 后面 arctanh 返回的绝对值太大了，做稍微裁剪
        eps = 1e-6
        old_action = torch.clamp(old_action, -1 + eps, 1 - eps)

        mean, std = self.forward(state)
        assert (torch.abs(old_action) < 1).all(), f'invalid action:{old_action}'
        normal = torch.distributions.Normal(mean, std)
        unscaled_action = torch.arctanh(old_action)

        log_prob = normal.log_prob(unscaled_action)
        # This is the crucial part, known as the "log-derivative of the tanh transformation" or
        # the "correction term for squashing functions".
        # It help to get the log_prob of the squashed action (action) in the squashed space
        log_prob -= torch.log(1 - old_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # todo：如果任务的action幅度不是+-1，还需要乘，并调整log_prob
        return  log_prob




class AggreVateDAgent:
    def __init__(self):
        self.env = gym.make(Config.env_name, render_mode=None)  # 开启可视化

        self.policy = PolicyNetwork(Config.state_dim, Config.action_dim, Config.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.lr)

        # 加载专家模型，来自SB3存放在hugging face上的预训练模型
        self.expert = SAC.load('./rl-trained-agents/sac/BipedalWalkerHardcore-v3_1/BipedalWalkerHardcore-v3.zip')
        self.alpha = 1.0

    def mixed_policy(self, state:torch.Tensor):
        if random.random() < self.alpha:
            obs = state.squeeze().cpu().numpy()
            action, _ = self.expert.predict(obs, deterministic=True)
        else:
            action, _ = self.policy.sample(state)
            action = action.squeeze(0).cpu().numpy()
        return action

    def decay_alpha(self, epoch):
        slope = 1.0 / Config.max_iteration
        self.alpha = 1.0 - slope * epoch
        return self.alpha


    def normalize_q_star(self, transition_list):
        q_star_list = []

        for timestep in transition_list:
            state, action, next_state, reward, done, q_star = timestep
            q_star_list.append(q_star)
        q_star_list = np.array(q_star_list) # type:np.ndarray
        q_star_list = (q_star_list - q_star_list.min()) / (q_star_list.max() - q_star_list.min() + 1e-8)
        return q_star_list

    def update(self, transition_list:list, normalized_q_star_list:np.ndarray, epoch:int):
        total_loss = 0.0
        loss_cnt = 0

        states_list, actions_list, _, _, _, _ = zip(*transition_list)
        for start in range(0, len(transition_list), Config.batch_sz):
            end = min(start + Config.batch_sz, len(transition_list) )
            state = states_list[start:end]
            action = actions_list[start:end]
            q_star = normalized_q_star_list[start:end]


            obs_tensor = torch.tensor(state).float().to(device)
            action_tensor = torch.tensor(action).float().to(device)
            q_star_tensor = torch.tensor(q_star).float().unsqueeze(-1).to(device)


            log_prob = self.policy.get_log_prob(obs_tensor, action_tensor)
            loss = -log_prob * q_star_tensor
            loss = loss.mean()
            total_loss += loss
            loss_cnt += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss / (loss_cnt + 1e-8)


    def evaluate(self, mode=None):
        env = gym.make(Config.env_name, render_mode=mode)
        total_reward = 0
        for _ in range(5):
            state, _ = env.reset()
            for _ in range(1000):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = self.policy.predict(state_tensor)
                    action = action.squeeze(0).cpu().numpy()

                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                state = next_state

                if done:
                    break

        env.close()
        return total_reward / 5

    def get_q_star(self, state, action):
        obs_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
        action_tensor = torch.tensor(action).float().unsqueeze(0).to(device)
        q1, q2 = self.expert.critic.forward(obs_tensor, action_tensor)
        q_star = torch.min(q1, q2)  # type:torch.Tensor
        q_star = q_star.squeeze(0)
        return q_star.item()

    def train(self):

        update_cnt = 0
        for epoch in tqdm(range(0, Config.max_iteration), 'trainning'):
            transition_list = []
            for _ in range(Config.K):
                state,_ = self.env.reset()
                for _ in range(Config.H):
                    state_tensor = torch.FloatTensor(state).to(device)
                    state_tensor = state_tensor.unsqueeze(0)
                    with torch.no_grad():
                        action = self.mixed_policy(state_tensor) #type:torch.Tensor
                    next_state, reward, done, _, _ = self.env.step(action)
                    q_star = self.get_q_star(state, action)

                    transition_list.append(
                                 (state,
                                  action,
                                  next_state,
                                  reward,
                                  done,
                                  q_star)
                                 )

                    state = next_state
                    if done:
                        break


            q_star_list = self.normalize_q_star(transition_list)  # type:np.ndarray
            for _ in range(Config.train_repeat):
                update_cnt += 1
                loss = self.update(transition_list, q_star_list,epoch)
                writer.add_scalar('train/loss', loss.item(), update_cnt)

            score = self.evaluate()
            writer.add_scalar('eval/episode_reward', score, epoch)

            self.decay_alpha(epoch)
            writer.add_scalar('train/alpha', self.alpha, epoch)

        self.evaluate('human')

# 6. 主函数（命令行参数解析）
# ----------------------------
def main(mode):
    agent = AggreVateDAgent()
    agent.train()


if __name__ == "__main__":
    main("train")
```


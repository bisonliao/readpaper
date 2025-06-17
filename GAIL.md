

**Generative Adversarial Imitation Learning**

### Introduction

先了解一下预备知识和相关概念

在强化学习（RL）中，策略是根据一个“奖励函数”或“代价函数”优化出来的。这个函数告诉 agent 哪些行为好、哪些不好。

恢复专家的代价函数，意思是：你观察一个专家（比如人类司机、下棋高手、机器人的演示轨迹），并试图推断出那个让专家选择这些行为的代价函数。

这个 cost 函数是通过 IRL 算法学到的，它解释了专家为什么选择某些动作而不是其他动作。逆向强化学习（IRL）是这样一种过程：给定专家的演示行为（比如状态-动作轨迹），你要反推出他们是根据什么代价函数做决策的。

IRL 的步骤通常是：

1. 输入：专家的轨迹（状态序列 + 动作序列）。
2. 优化：寻找一个 cost function，使得专家的行为在这个代价下是最优的。
3. 然后，你可以用普通 RL 方法（如 TRPO、PPO）在这个 cost function 上优化出一套自己的策略。



论文提出了一个新的通用框架，可以直接从数据中提取策略。将模仿学习与生成对抗网络（GAN）进行类比。基于这一类比，我们设计出一个无需环境模型的模仿学习算法，它在各种复杂、高维环境中模仿专家行为时，性能明显优于现有的无模型方法。



The learner is given only samples of trajectories from the expert, is not allowed to query the expert for more data while training, and is not provided
reinforcement signal of any kind. 

There are two main approaches suitable for this setting: 

1. behavioral cloning, which learns a policy as a supervised learning problem over state-action pairs from expert trajectories; 
2. inverse reinforcement learning, which finds a cost function under which the expert is uniquely optimal.

Our characterization introduces a framework for directly learning policies from data, bypassing any intermediate IRL step.

### Background

论文在这部分介绍了IRL什么的，我倒是觉得需要帮我复习一下GAN

![image-20250616113905927](img/image-20250616113905927.png)

### 诱导最优策略的表征

看不太懂，大概摘录一下我认为重要的内容：

![image-20250616115802793](img/image-20250616115802793.png)



### 实用的占用度量匹配

![image-20250616122201314](img/image-20250616122201314.png)

### 生成对抗模仿学习（GAIL）

![image-20250616122421532](img/image-20250616122421532.png)

#### 说人话

```python
# 初始化策略 πθ 和判别器 Dω
initialize_policy_network()
initialize_discriminator_network()

# 准备专家数据 (s, a) 对
expert_dataset = load_expert_trajectories()

for iteration in range(num_iterations):

    # === 1. 用当前策略 πθ 跑环境，采样一批 trajectory ===
    fake_trajectories = rollout_policy(policy=πθ, env, num_steps)
    fake_batch = extract_state_action_pairs(fake_trajectories)

    # === 2. 训练判别器 Dω，判断 expert / agent ===
    for _ in range(num_discriminator_steps):
        expert_batch = sample_batch(expert_dataset)
        fake_batch = sample_batch(fake_batch)

        # 判别器 loss: 二分类交叉熵
        D_loss = -mean(log(D(s,a)) for (s,a) in expert_batch) \
                 -mean(log(1 - D(s,a)) for (s,a) in fake_batch)

        update_discriminator(D_loss)

    # === 3. 把 log D(s, a) 当作 reward，训练策略 πθ ===
    for _ in range(num_policy_update_steps):

        # 用 log D(s, a) 当 reward，计算 return / advantage
        rewards = [log(D(s, a)) for (s, a) in fake_trajectories]
        returns, advantages = estimate_advantage(rewards)

        # 用 PPO / REINFORCE / TRPO 更新策略
        policy_loss = -log_prob * advantage (with clipping if PPO)
        update_policy(policy_loss)

    # === 4. 日志输出（可选）===
    print(f"Iter {iteration}: avg_reward = ..., D_acc = ...")

```

![image-20250616171729899](img/image-20250616171729899.png)

#### fake reward绝对值很大怎么办

用log D(s,a)做为reward，负值的绝对值可能很大，这种情况下推荐怎么做？还是说不需要额外处理？

![image-20250616170756265](img/image-20250616170756265.png)

#### 输入只是(s,a)数据对？

![image-20250616123835214](img/image-20250616123835214.png)

#### GAIL一定要与环境交互吗？

![image-20250616142633232](img/image-20250616142633232.png)

#### GAIL只能训练策略网络吗？

![image-20250616170020021](img/image-20250616170020021.png)

#### cost函数就是一个度量指标

cost函数这个概念感觉有点奇奇怪怪的：

![image-20250616135959370](img/image-20250616135959370.png)

### Experiments

![image-20250616141143161](img/image-20250616141143161.png)

### bison的实验

#### cartpole（结果不错）

实验设计：

1. 训练一个DQN作为专家，与CartPole环境交互，得到一系列离线的专家经验
2. 使用GAIL训练G和D，RL算法采用REINFORCE算法

结果：能学到不错的策略，show_case的输出：

```shell
episode len:500
episode len:500
episode len:500
episode len:500
episode len:500
episode len:500
episode len:500
episode len:500
episode len:500
episode len:500
```

![image-20250617104211983](img/image-20250617104211983.png)

```python
import datetime
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1", render_mode=None, max_episode_steps=500)
n_state = env.observation_space.shape[0]  # 状态维度
n_action = env.action_space.n  # 动作数量
writer = SummaryWriter(log_dir=f'./logs/GAIL_{datetime.datetime.now().strftime("%m%d_%H%M%S")}')


# DQN 网络定义，作为专家提供经验
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# GAIL训练的策略网络，也是生成器
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

#判别器定义
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, states, actions):
        actions_one_hot = F.one_hot(actions.to(torch.long), num_classes=n_action)
        x = torch.concatenate([states, actions_one_hot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x) # 0-1的概率

# 待训练的策略网络与环境交互，获得rollout数据
def interact_with_env(g:Policy):
    fake_samples = []
    traj_lens = []
    g.eval()
    while len(fake_samples) < 2048:
        state = env.reset()
        state = state[0]
        ep_len = 0
        for _ in range(500):
            with torch.no_grad():
                stateTensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = g(stateTensor) # type:torch.Tensor
                dist = Categorical(action)
                action = dist.sample().item()
                #action = action.argmax(dim=1)[0].cpu().item()

            fake_samples.append((state, action))
            ep_len += 1
            state, reward, done, _, _ = env.step(action)

            if done:
                traj_lens.append(ep_len)
                break
        if not done: #不是因为终止而退出上述循环
            traj_lens.append(ep_len)
    g.train()
    return fake_samples, traj_lens

# 根据奖励计算累计回报
def compute_returns(rewards, traj_lens:list, gamma=0.99):
    returns = []
    # assert sum(traj_lens) == rewards.shape[0], f"sum {traj_lens}  != {rewards.shape[0]}"
    # split rewards as trajectory
    reward_seg = []
    start = 0
    for l in traj_lens:
        end = start + l
        reward_seg.append( rewards[start:end] )
        start = end

    # calc returns for each segment
    for seg in reward_seg:
        R = 0
        returns_seg = []
        for r in reversed(seg):
            R = r + gamma * R
            returns_seg.insert(0, R)
        returns.extend(returns_seg)

    return torch.tensor(returns, device=device)

#利用样本数据，REINFORCE算法更新策略网络
def update_policy_network(policy, optimizer, states, actions, returns):

    # 1. 通过策略网络计算动作概率
    probs = policy(states)  # [T, action_dim]
    # 2. 创建分类分布（用于采样和计算对数概率）
    m = Categorical(probs)
    # 3. 计算所选动作的对数概率
    log_probs = m.log_prob(actions)  # [T,]
    # 4. 因为基于策略的强化学习要使用梯度上升使得state-value函数的期望最大化，所以损失函数是期望值的负数
    # returns已经在函数外面进行了带折扣的汇总运算，也就是已经是U了，不是每一步的r
    loss = -(log_probs * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def show_case(policy:Policy):
    eval_env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)
    policy.eval()
    for _ in range(10):
        state = eval_env.reset()
        state = state[0]
        ep_len = 0
        for _ in range(500):
            with torch.no_grad():
                stateTensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy(stateTensor)  # type:torch.Tensor
                action = action.argmax(dim=1)[0].cpu().item()
            ep_len += 1
            state, reward, done, _, _ = eval_env.step(action)

            if done:
                break
        print(f"episode len:{ep_len}")
    policy.train()


# 利用GAIL算法训练策略网络
def GAIL_train_policy(experiments_file='./expert_trajectory.pth'):
    max_epoches = 800
    lr_g = 1e-4
    lr_d = 1e-4
    batch_size = 128
    gamma = 0.99

    true_samples = torch.load(experiments_file, weights_only=False) #type:list
    generator = Policy(n_state, n_action).to(device)
    discriminator = Discriminator(n_state, n_action).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)

    for ep in range(max_epoches):
        fake_samples, traj_lens = interact_with_env(generator)

        #训练判别器
        for _ in range(4):
            batch_size_hf = batch_size // 2
            inputs = random.sample(true_samples, batch_size_hf) + random.sample(fake_samples, batch_size_hf)
            labels = [0] * batch_size_hf + [1] * batch_size_hf
            # 1. 打包成 (input, label) 对
            combined = list(zip(inputs, labels))
            # 2. shuffle 整体顺序
            random.shuffle(combined)
            # 3. 解包回来
            inputs, labels = zip(*combined)
            states, actions = zip(*inputs)

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            outputs = discriminator(states, actions) #输出的是(B,)的0-1概率， 1表示是专家
            d_loss = nn.functional.binary_cross_entropy(outputs.squeeze(1), labels.float())
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

        #训练生成器
        for _ in range(2):
            inputs = fake_samples
            states, actions = zip(*inputs)

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = discriminator(states, actions)# type:torch.Tensor
            rewards = -torch.log( 1 - outputs + (1e-8) ) # 1e-8, 一方面防止0导致无穷小，一方面防止reward绝对值太大
            # 计算回报
            returns = compute_returns(rewards, traj_lens, gamma)
            # 让权重有正有负，如果正的，我们就要增大在这个状态采取这个动作的概率；如果是负的，我们就要减小在这个状态采取这个动作的概率
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            # assert actions.shape == returns.shape
            # 更新策略
            g_loss = update_policy_network(generator, optimizer_g, states, actions, returns)

        writer.add_scalar('GAIL/d_loss', d_loss.item(), ep)
        writer.add_scalar('GAIL/g_loss', g_loss.item(), ep)
        writer.add_scalar('GAIL/rewards', rewards.mean().cpu().item(), ep)
    show_case(generator)


# 训练专家网络过程中，与环境交互
def select_action_from_expert(model, state, epsilon):
    """基于 ε-greedy 选择动作"""
    if random.random() < epsilon:
        return random.randint(0, n_action - 1)  # 随机选择
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 变换前：[4] -> 变换后：[1, 4]
        return model(state).argmax(1).item()  # 选取 Q 值最大的动作

# 训练专家网络过程中，更新其参数
def update_expert(model, target_model, memory, batch_size, gamma, optimizer):
    if len(memory) < batch_size:
        return 9999.0  # 经验池数据不足时不训练

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)  # (batch_size, 4)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
    next_states = torch.FloatTensor(next_states).to(device)  # (batch_size, 4)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)

    # 计算当前 Q 值
    q_values = model(states).gather(1, actions)  # 从 Q(s, a) 选取执行的动作 Q 值

    # 计算目标 Q 值
    next_q_values = target_model(next_states).max(1, keepdim=True)[0]  # 选取 Q(s', a') 的最大值
    target_q_values = rewards + gamma * next_q_values * (1 - dones)  # TD 目标

    # 计算损失
    loss = F.mse_loss(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 训练expert网络，并获得专家交互经验
def get_expert_trajectory():
    # 超参数
    gamma = 0.99  # 折扣因子
    epsilon = 1.0  # 初始探索率
    epsilon_min = 0.01  # 最低探索率
    epsilon_decay = 0.995  # 探索率衰减
    learning_rate = 1e-3  # 学习率
    batch_size = 128  # 经验回放的批量大小
    memory_size = 200_000  # 经验池大小
    target_update_freq = 10  # 目标网络更新频率
    episode_max_steps = 800

    # 初始化网络
    model = DQN(n_state, n_action).to(device)
    target_model = DQN(n_state, n_action).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)

    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        state = state[0]  # 适配 Gym v26
        total_reward = 0

        for _ in range(episode_max_steps):
            action = select_action_from_expert(model, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            # 经验回放缓存
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 训练 DQN
            loss = update_expert(model, target_model, memory, batch_size, gamma, optimizer)

            if done:
                break

        # 逐步降低 epsilon，减少随机探索，提高利用率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 定期更新目标网络，提高稳定性
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        writer.add_scalar('expert/episode_rew', total_reward, episode)
        writer.add_scalar('expert/epsilon', epsilon, episode)
        writer.add_scalar('expert/loss', loss, episode)
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, loss:{loss}")

    # generate expert trajectory
    expert_experience = []
    model.eval()
    while len(expert_experience) < 200_000:
        state = env.reset()
        state = state[0]
        total_reward = 0



        for _ in range(episode_max_steps):
            with torch.no_grad():
                action = select_action_from_expert(model, state, 0)  # 纯利用，epsilon=0
            expert_experience.append( (state, action) )
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                break
        print(f"Total reward: {total_reward}")
        writer.add_scalar('expert/experience', total_reward, len(expert_experience))

    torch.save(expert_experience, './expert_trajectory.pth')


if __name__ == "__main__":
    #get_expert_trajectory()
    GAIL_train_policy()

```

#### 完全离线的变种GAIL

完全不与环境交互，只用 专家数据（expert trajectories）作为正样本**，**生成器采样 fake 样本，通过 GAN 式对抗训练生成器。

这本质上是一个 **“行为克隆 + GAN”** 的 GAIL 简化版本，也可以理解为 **offline GAIL / imitation learning**。我们用 policy 网络采样动作，并用专家数据+生成器数据训练判别器，再通过判别器的 reward 更新 policy。

结果：

也能学习到一定的能力，但加大训练的epoches的时候，能力反而下降了。

下面是show_case的输出

```shell
episode len:172
episode len:189
episode len:157
episode len:281
episode len:229
episode len:186
episode len:205
episode len:216
episode len:262
episode len:203
```

![image-20250617113829881](img/image-20250617113829881.png)



```python
import datetime
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1", render_mode=None, max_episode_steps=500)
n_state = env.observation_space.shape[0]  # 状态维度
n_action = env.action_space.n  # 动作数量
writer = SummaryWriter(log_dir=f'./logs/GAIL_offline_{datetime.datetime.now().strftime("%m%d_%H%M%S")}')


# DQN 网络定义，作为专家提供经验
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# GAIL训练的策略网络，也是生成器
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

#判别器定义
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, states, actions):
        actions_one_hot = F.one_hot(actions.to(torch.long), num_classes=n_action)
        x = torch.concatenate([states, actions_one_hot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x) # 0-1的概率


def show_case(policy:Policy):
    eval_env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)
    policy.eval()
    for _ in range(10):
        state = eval_env.reset()
        state = state[0]
        ep_len = 0
        for _ in range(500):
            with torch.no_grad():
                stateTensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy(stateTensor)  # type:torch.Tensor
                action = action.argmax(dim=1)[0].cpu().item()
            ep_len += 1
            state, reward, done, _, _ = eval_env.step(action)

            if done:
                break
        print(f"episode len:{ep_len}")
    policy.train()


# 利用GAIL算法训练策略网络
def GAIL_train_policy(
    experiments_file='./expert_trajectory.pth',
    max_epochs=500,
    d_steps=5,
    g_steps=1,
    batch_size=128,
    gamma=0.99,
    lr_d=1e-4,
    lr_g=1e-4
):
    # 加载专家数据 [(state, action)]
    expert_data = torch.load(experiments_file, weights_only=False)
    expert_states, expert_actions = zip(*expert_data)
    expert_states = torch.tensor(expert_states, dtype=torch.float32).to(device)
    expert_actions = torch.tensor(expert_actions, dtype=torch.long).to(device)

    # 初始化网络
    generator = Policy(n_state, n_action).to(device)
    discriminator = Discriminator(n_state, n_action).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)

    for ep in range(max_epochs):
        # ============ 1. 判别器训练 ============
        for _ in range(d_steps):
            idx_exp = torch.randint(0, expert_states.shape[0], (batch_size,))
            states_exp = expert_states[idx_exp]
            actions_exp = expert_actions[idx_exp]

            idx_fake = torch.randint(0, expert_states.shape[0], (batch_size,))
            states_fake = expert_states[idx_fake]
            with torch.no_grad():
                probs = generator(states_fake)
                dist = Categorical(probs)
                actions_fake = dist.sample()

            # 构造混合数据
            states_all = torch.cat([states_exp, states_fake], dim=0)
            actions_all = torch.cat([actions_exp, actions_fake], dim=0)
            labels = torch.cat([
                torch.ones(batch_size),  # expert: 1
                torch.zeros(batch_size)  # fake: 0
            ]).to(device)

            preds = discriminator(states_all, actions_all).squeeze()
            loss_d = F.binary_cross_entropy(preds, labels)
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        # ============ 2. 生成器训练 ============
        for _ in range(g_steps):
            idx_fake = torch.randint(0, expert_states.shape[0], (batch_size,))
            states_fake = expert_states[idx_fake]

            with torch.no_grad():
                # === rollout ===
                old_probs = generator(states_fake)
                old_dist = Categorical(old_probs)
                old_actions = old_dist.sample()

                # === reward ===
                d_pred = discriminator(states_fake, old_actions).squeeze()
                rewards = -torch.log(1 - d_pred + 1e-8)

            # === 训练阶段，用相同的动作 ===
            new_probs = generator(states_fake)
            new_dist = Categorical(new_probs)
            log_probs = new_dist.log_prob(old_actions)  # ✅ 注意：不是重新采样

            # === 策略梯度更新 ===
            returns = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
            loss_g = -(log_probs * returns).mean()
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        # 日志记录
        writer.add_scalar('GAIL2/d_loss', loss_d.item(), ep)
        writer.add_scalar('GAIL2/g_loss', loss_g.item(), ep)
        writer.add_scalar('GAIL2/reward', rewards.mean().item(), ep)

        if ep % 20 == 0 or ep == max_epochs - 1:
            print(f"[Epoch {ep}] d_loss: {loss_d.item():.4f} | g_loss: {loss_g.item():.4f} | avg_reward: {rewards.mean().item():.4f}")

    # 可视化展示
    show_case(generator)




# 训练专家网络过程中，与环境交互
def select_action_from_expert(model, state, epsilon):
    """基于 ε-greedy 选择动作"""
    if random.random() < epsilon:
        return random.randint(0, n_action - 1)  # 随机选择
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 变换前：[4] -> 变换后：[1, 4]
        return model(state).argmax(1).item()  # 选取 Q 值最大的动作

# 训练专家网络过程中，更新其参数
def update_expert(model, target_model, memory, batch_size, gamma, optimizer):
    if len(memory) < batch_size:
        return 9999.0  # 经验池数据不足时不训练

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)  # (batch_size, 4)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
    next_states = torch.FloatTensor(next_states).to(device)  # (batch_size, 4)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)

    # 计算当前 Q 值
    q_values = model(states).gather(1, actions)  # 从 Q(s, a) 选取执行的动作 Q 值

    # 计算目标 Q 值
    next_q_values = target_model(next_states).max(1, keepdim=True)[0]  # 选取 Q(s', a') 的最大值
    target_q_values = rewards + gamma * next_q_values * (1 - dones)  # TD 目标

    # 计算损失
    loss = F.mse_loss(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 训练expert网络，并获得专家交互经验
def get_expert_trajectory():
    # 超参数
    gamma = 0.99  # 折扣因子
    epsilon = 1.0  # 初始探索率
    epsilon_min = 0.01  # 最低探索率
    epsilon_decay = 0.995  # 探索率衰减
    learning_rate = 1e-3  # 学习率
    batch_size = 128  # 经验回放的批量大小
    memory_size = 200_000  # 经验池大小
    target_update_freq = 10  # 目标网络更新频率
    episode_max_steps = 800

    # 初始化网络
    model = DQN(n_state, n_action).to(device)
    target_model = DQN(n_state, n_action).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)

    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        state = state[0]  # 适配 Gym v26
        total_reward = 0

        for _ in range(episode_max_steps):
            action = select_action_from_expert(model, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            # 经验回放缓存
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 训练 DQN
            loss = update_expert(model, target_model, memory, batch_size, gamma, optimizer)

            if done:
                break

        # 逐步降低 epsilon，减少随机探索，提高利用率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 定期更新目标网络，提高稳定性
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        writer.add_scalar('expert/episode_rew', total_reward, episode)
        writer.add_scalar('expert/epsilon', epsilon, episode)
        writer.add_scalar('expert/loss', loss, episode)
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, loss:{loss}")

    # generate expert trajectory
    expert_experience = []
    model.eval()
    while len(expert_experience) < 200_000:
        state = env.reset()
        state = state[0]
        total_reward = 0



        for _ in range(episode_max_steps):
            with torch.no_grad():
                action = select_action_from_expert(model, state, 0)  # 纯利用，epsilon=0
            expert_experience.append( (state, action) )
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                break
        print(f"Total reward: {total_reward}")
        writer.add_scalar('expert/experience', total_reward, len(expert_experience))

    torch.save(expert_experience, './expert_trajectory.pth')


if __name__ == "__main__":
    #get_expert_trajectory()
    GAIL_train_policy()

```


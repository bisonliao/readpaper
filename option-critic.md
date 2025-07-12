**The Option-Critic Architecture**

### 1、Introduction

1. **Temporal abstraction** 是强化学习中提升学习效率与计划能力的关键，它通过定义跨时间尺度的行为序列（options）来实现更高层次的动作选择。
2. 尽管规划方面已经掌握了如何使用 temporally extended actions（例如 options），但 **如何自动地从数据中学习这些抽象层次** 依然是一个挑战。
3. 传统方法偏向于先发现 subgoals（中间目标状态），再学习到达这些 subgoals 的策略。这种方法难以扩展到更大、更复杂的环境：
   - 寻找 subgoals 是组合性问题，计算成本高。
   - 学习每个子策略的代价可能不比直接解决整个任务低多少。
4. 论文提出一种 **无需 subgoal 指定、奖励重构或额外问题设计** 的方法，直接从经验中学习 options。所提出的 **option-critic 架构** 同时学习以下三个关键组件：
   1. intra-option policies（option 内部策略）
   2. termination functions（option 的终止条件）
   3. policy over options（选择哪个 option 的元策略）
5. 论文的方法基于 policy gradient 原理，**可端到端优化最终任务目标**，且适用于离散或连续空间、线性或非线性函数逼近器。
6. 该方法不仅能在单任务中高效学习 options，还能在迁移任务中展现较好的泛化能力。
7. 在多个环境（包括经典控制任务和 Atari 游戏）中验证其性能与效率

在强化学习中，“**temporal abstraction（时间抽象）**”是一种让智能体不用每一步都做底层决策、而是能在更长的时间尺度上思考和行动的方式。可以把它理解为我们人类在计划时不去纠结每个细节动作，而是用**高层次的行为块**来组织任务。

在强化学习中，**Options** 就是实现时间抽象的方式：

- **一个 option =（起始条件，内部策略，终止条件）**
- agent 在某个状态选择一个 option，它会沿着内部策略执行多个时间步，直到满足终止条件
- 执行期间 agent 不会做原始动作决策，而是“委托”option 控制

### 2、Preliminaries and Notation

介绍了MDP、Policy gradient methods、The options framework。

![image-20250709142906386](img/image-20250709142906386.png)

### 3、Learning Options

懵懵懂懂的，没有搞明白

![image-20250709145637007](img/image-20250709145637007.png)

### 4、Algorithms and Architecture

#### 4.1 算法伪代码

![image-20250709151338087](img/image-20250709151338087.png)

```
# 初始化神经网络参数
initialize_shared_encoder()          # 表征状态 s 的共享 encoder，例如 CNN 或 MLP
initialize_option_policies()         # 每个 option w 的 π_wθ(a|s)，可共用 encoder + 分别的 head
initialize_termination_heads()       # 每个 option w 的 β_wϕ(s)：shared encoder → FC → sigmoid
initialize_policy_over_options()     # 元策略 π_Ω(w|s)，可选 softmax 或 ε-greedy
initialize_critic_Q_U_and_Q_Ω()      # 初始化 Q_U(s, w, a) 和 Q_Ω(s, w)

# 环境初始化
s = env.reset()
features = shared_encoder(s)
w = sample_from_policy_over_options(features)  # 选择初始 option

while not done:

    # 1. 从当前 option 的策略中采样动作
    a = sample_from_intra_option_policy(w, features)  # π_wθ(a | s)

    # 2. 执行动作，与环境交互
    s_next, r, done = env.step(a)
    features_next = shared_encoder(s_next)

    # 3. TD 目标用于 critic 更新
    target = r + γ * (
        (1 - β_wϕ(s_next)) * Q_Ω(s_next, w)
      + β_wϕ(s_next)      * max_w′ Q_Ω(s_next, w′)
    )

    # 4. 更新 critic 网络（可使用 TD 或 Q-learning）
    update_Q_U(s, w, a, target)
    Q_Ω(s, w) = estimate_Q_Ω_from_Q_U(s, w)

    # 5. 更新 intra-option policy（策略头）
    # 使用 Q_U(s,w,a) 指导策略提升 log π_w(a|s)
    update_policy_head(w, s, a, Q_U[s,w,a])

    # 6. 更新 termination function β_wϕ(s)
    # 使用优势函数指导其更倾向终止或延续
    A = Q_Ω(s_next, w) - V_Ω(s_next)  # 可选 baseline
    update_termination_head(w, s_next, advantage=A)  # ∇ϕ β_wϕ(s) · A

    # 7. 决定是否退出当前 option
    β = termination_head(w, features_next)  # forward 神经网络，输出终止概率
    if random() < β:
        w = sample_from_policy_over_options(features_next)  # 重新选 option

    # 8. 前进一步
    s = s_next
    features = features_next

```

#### 4.2 深入理解

##### 实现表示

![image-20250709155203486](img/image-20250709155203486.png)

![image-20250709164855973](img/image-20250709164855973.png)

##### 三个HRL算法的对比

![image-20250709160533200](img/image-20250709160533200.png)

### 5、Experiments

![image-20250709161507525](img/image-20250709161507525.png)

### 6、Related Work

### 7、Discussion

   Option-Critic 提供了一个基于策略梯度的统一架构，能同时学习：

- 每个 option 的内部策略（how to act）
- 每个 option 的终止函数（when to stop）
- 高层策略（which option to pick）

且无需额外的 subgoal、伪奖励或人工结构——**只需指定 option 数量即可**。

虽然原始方法只使用环境奖励，但也支持引入伪奖励

可以利用额外奖励鼓励 option 更具时间延展性（例如通过 penalize 频繁切换）

只要奖励结构是“可加”的形式（additive），都可以无缝融合

论文承认：基于折扣回报的策略梯度是**有偏的**（Thomas, 2014）

尽管可以构造无偏估计器，但其样本复杂度太高

作者在实验中发现：**有偏估计器也能表现良好，实用性更高**

当前假设每个 option 在所有状态都能被执行（universal initiation）

若引入 initiation set（option 起始集合）的函数逼近，会增加计算复杂度

不过作者指出这类机制是可以学习的，只是留作未来工作推进

### 8、bison的实验

##### MountainCar

没有搞定。



论文作者提供了官方代码：

```
https://github.com/jeanharb/option_critic
```

AI帮我写代码如下：

```python
import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import random
import time

class NormalizeWrapper(gym.Wrapper):
    def __init__(self, env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0, epsilon=1e-8):
        super().__init__(env)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.epsilon = epsilon

        obs_shape = self.observation_space.shape
        self.obs_rms_count = 0
        self.obs_mean = np.zeros(obs_shape, dtype=np.float32)
        self.obs_var = np.ones(obs_shape, dtype=np.float32)

        self.ret_rms_count = 0
        self.ret_mean = 0.0
        self.ret_var = 1.0

        self.ret = 0.0  # running return
        self.training = True  # if False, stop updating statistics

    def reset(self, **kwargs):
        self.ret = 0.0
        obs, info = self.env.reset(**kwargs)
        if self.norm_obs:
            self._update_obs_rms(obs)
            obs = self._normalize_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if self.norm_obs:
            self._update_obs_rms(obs)
            obs = self._normalize_obs(obs)

        if self.norm_reward:
            self.ret = self.ret * self.env.spec.reward_threshold + reward
            self._update_ret_rms(self.ret)
            reward = self._normalize_reward(reward)

        if done:
            self.ret = 0.0

        return obs, reward, terminated, truncated, info

    def _update_obs_rms(self, obs):
        if not self.training:
            return
        self.obs_rms_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_rms_count
        self.obs_var += (obs - self.obs_mean) * delta

    def _update_ret_rms(self, ret):
        if not self.training:
            return
        self.ret_rms_count += 1
        delta = ret - self.ret_mean
        self.ret_mean += delta / self.ret_rms_count
        self.ret_var += (ret - self.ret_mean) * delta

    def _normalize_obs(self, obs):
        std = np.sqrt(self.obs_var / (self.obs_rms_count + 1e-8)) + self.epsilon
        obs_normalized = (obs - self.obs_mean) / std
        return np.clip(obs_normalized, -self.clip_obs, self.clip_obs)

    def _normalize_reward(self, reward):
        std = np.sqrt(self.ret_var / (self.ret_rms_count + 1e-8)) + self.epsilon
        reward_normalized = (reward - self.ret_mean) / std
        return np.clip(reward_normalized, -self.clip_reward, self.clip_reward)


# ==================== 超参数集中配置 ====================
class Args:
    env_id = "MountainCar-v0"
    num_options = 4
    total_steps = 1_000_000
    update_freq = 2048
    eval_interval = 100
    max_ep_len = 200
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    vf_coef = 0.5
    term_reg = 0.01
    ppo_epochs = 10
    batch_size = 256
    lr = 3e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 42

args = Args()
writer = SummaryWriter(f'logs/OptionCritic_MountainCar_{datetime.datetime.now().strftime("%m%d_%H%M%S")}')

# ==================== 环境和种子 ====================
env = gym.make(args.env_id)
env = NormalizeWrapper(env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

eval_env = gym.make(args.env_id)
eval_env = NormalizeWrapper(eval_env)

# ==================== 模块定义 ====================
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.fc(x)

class PPOOptionHead(nn.Module):
    def __init__(self, hidden_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, act_dim)
        )
        self.critic = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class Termination(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)

class EpsilonScheduler:
    def __init__(self, start=1.0, end=0.1, decay_steps=Args.total_steps / 2):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def get(self, current_step):
        epsilon = self.start - (self.start - self.end) * min(1.0, current_step / self.decay_steps)
        return epsilon
# ==================== 存储器 ====================
class TrajectoryBuffer:
    def __init__(self):
        self.reset()
    def store(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)
    def get(self):
        return {k: torch.stack(v) for k, v in self.data.items()}
    def reset(self):
        self.data = {k: [] for k in ['obs','action','option','logp','reward','value','done','terminate','mask','hidden','term_prob']}


# ==================== 主体结构 ====================
class OptionCriticAgent:
    def __init__(self):
        self.encoder = SharedEncoder(obs_dim).to(args.device)
        self.options = nn.ModuleList([
            PPOOptionHead(128, act_dim).to(args.device) for _ in range(args.num_options)
        ])
        self.terminations = nn.ModuleList([
            Termination(128).to(args.device) for _ in range(args.num_options)
        ])

        # 新增：Q(s, o) 网络，用于高层 option 策略（ε-greedy）
        self.q_option_head = nn.Linear(128, args.num_options).to(args.device)
        self.q_option_optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.q_option_head.parameters()),
                                               lr=args.lr)

        self.optims = []
        for i in range(args.num_options):
            self.optims.append( torch.optim.Adam(list(self.options[i].parameters()) + list(self.terminations[i].parameters()), lr=args.lr))

        self.epsilon_scheduler = EpsilonScheduler()
        self.update_cnt = 0

    # 进行低层微观动作
    def act(self, obs, option:int, eval_mode=False):
        if not hasattr(self, "act_cnt"):
            self.act_cnt = 0  # 初始化
        self.act_cnt += 1

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(args.device)
        with torch.no_grad():
            h = self.encoder(obs)
            logits, value = self.options[option](h)
            dist = Categorical(logits=logits)
            action = dist.sample() if not eval_mode else dist.probs.argmax()
            logp = dist.log_prob(action)
            term_prob = self.terminations[option](h)

        if self.act_cnt % 100 ==0:
            writer.add_scalar('train/action', action.item(), self.act_cnt)
        return action.item(), logp, value.squeeze(), h.detach(), term_prob

    # 选择 option（高层策略 π_hi），返回 sampled option 和其 logp，用于 PG
    def choose_option(self, obs, eval_mode=False, epsilon=0.1):
        if not hasattr(self, "choose_option_cnt"):
            self.choose_option_cnt = 0  # 初始化
        self.choose_option_cnt += 1

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(args.device)
        with torch.no_grad():
            h = self.encoder(obs)
            q_o = self.q_option_head(h).squeeze(0)  # [num_options]
            if eval_mode or random.random() > epsilon:
                option = torch.argmax(q_o).item()
            else:
                option = random.randint(0, args.num_options - 1)
        if self.choose_option_cnt % 100 == 0:
            writer.add_scalar('train/choose_option', option, self.choose_option_cnt)
        return option

    def compute_gae(self, buffer: TrajectoryBuffer, next_state: np.ndarray, option: int):
        data = buffer.get()
        rewards = data['reward']
        values = data['value']
        masks = data['mask']

        # 为了接上 V_{T+1}，我们额外推一帧
        next_obs = torch.tensor(next_state, dtype=torch.float32).to(args.device)
        with torch.no_grad():
            h = self.encoder(next_obs)
            _, next_value = self.options[option](h)
        next_value = next_value.squeeze().detach()

        # 拼接 values: V_0, ..., V_T, V_{T+1}
        values = torch.cat([values, next_value.unsqueeze(0)])

        advantages = torch.zeros_like(rewards).to(args.device)
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + args.gamma * values[t + 1] * masks[t] - values[t]
            gae = delta + args.gamma * args.gae_lambda * masks[t] * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        advantages = (advantages-advantages.mean()) / (advantages.std()+1e-8)
        return advantages.detach(), returns.detach()

    def update(self, buffer: TrajectoryBuffer, next_obs: np.ndarray, option: int):
        data = buffer.get()
        rewards = data['reward']
        masks = data['mask']
        done_flags = data['done'].float()

        # ---------- Q(s, o) 目标估计 ----------
        with torch.no_grad():
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(args.device)
            h_next = self.encoder(next_obs_tensor)
            q_next = self.q_option_head(h_next).squeeze(0)  # Q(s_{T+1}, ·)
        q_targets = []

        for t in range(len(rewards)):
            o_t = data['option'][t]
            beta_t1 = data['term_prob'][t+1] if t+1 < len(data['term_prob']) else torch.tensor(1.0).unsqueeze(0).to(args.device)
            h_t1 = data['hidden'][t+1] if t+1 < len(data['hidden']) else h_next
            q_next_all = self.q_option_head(h_t1).squeeze(0).detach()
            q_next_max = q_next_all.max()

            q_val = rewards[t] + args.gamma * (
                (1 - beta_t1) * q_next_all[o_t] + beta_t1 * q_next_max
            ) * masks[t]
            q_targets.append(q_val)

        q_targets = torch.stack(q_targets)
        h_batch = data['hidden']
        o_batch = data['option']
        q_pred = self.q_option_head(h_batch)[range(len(o_batch)), o_batch]
        loss_q = F.mse_loss(q_pred, q_targets.detach())
        self.q_option_optim.zero_grad()
        loss_q.backward()
        self.q_option_optim.step()
        writer.add_scalar('train/hi_q_loss', loss_q.item(), self.update_cnt)

        # ---------- Option-level PPO 更新 ----------
        adv, ret = self.compute_gae(buffer, next_obs, option)
        loss_list = []
        for option_id in range(args.num_options):
            idx = (data['option'][:-1] == option_id)
            next_idx = (data['option'][1:] == option_id)

            if idx.sum() == 0 and next_idx.sum() == 0:
                continue

            obs = data['obs'][:-1][idx]
            action = data['action'][:-1][idx]
            logp_old = data['logp'][:-1][idx]
            adv_opt = adv[:-1][idx]
            ret_opt = ret[:-1][idx]
            h = data['hidden'][:-1][idx].detach()

            logits, value = self.options[option_id](h)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(action)
            ratio = (logp - logp_old).exp()
            pg_loss = -torch.mean(torch.min(
                ratio * adv_opt,
                torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_opt
            ))
            v_loss = F.mse_loss(value.squeeze(), ret_opt)
            entropy = dist.entropy().mean()

            h_next = data['hidden'][1:][next_idx]  # 注意是 t+1 时刻的 hidden state
            beta = self.terminations[option_id](h_next)
            term = data['terminate'][1:][next_idx].float()
            advantage_term = (data['value'][1:][next_idx] - data['value'][1:][next_idx].max()).detach()
            term_loss = -torch.mean(term * torch.log(beta + 1e-6) * advantage_term +
                                    (1 - term) * torch.log(1 - beta + 1e-6) * advantage_term)

            loss = pg_loss + args.vf_coef * v_loss - args.entropy_coef * entropy + args.term_reg * term_loss
            loss_list.append(loss.item())
            self.optims[option_id].zero_grad()
            loss.backward()
            self.optims[option_id].step()
        writer.add_scalar('train/lo_loss_mean', np.mean(np.array(loss_list)), self.update_cnt)
        buffer.reset()
        self.update_cnt += 1

    def evaluate(self,  total_steps):
        eval_reward = 0
        eval_steps = 0
        obs, _ = eval_env.reset()
        option = self.choose_option(obs, eval_mode=True)

        while eval_steps < args.max_ep_len:
            action, _, _, _, term_prob = self.act(obs, option, eval_mode=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            eval_reward += reward
            eval_steps += 1

            should_terminate = torch.bernoulli(term_prob).bool()
            if should_terminate and not done:
                option = self.choose_option(obs, eval_mode=True)

            if done:
                break
        writer.add_scalar("eval/episode_return", eval_reward, total_steps)


# ==================== 主训练循环 ====================
def train():
    agent = OptionCriticAgent()
    buffer = TrajectoryBuffer()
    epsilon = agent.epsilon_scheduler.get(0)
    obs, _ = env.reset()
    option = agent.choose_option(obs, epsilon=epsilon)
    total_steps = 0
    ep_reward = 0
    episode = 0
    ep_return_list = []

    while total_steps < args.total_steps:
        epsilon = agent.epsilon_scheduler.get(total_steps)
        writer.add_scalar('train/epsilon', epsilon, total_steps)
        for _ in range(args.update_freq): #PPO Style，收集固定步数的transition
            action, logp, value, h, term_prob = agent.act(obs, option) #低层策略输出微观动作等
            next_obs, reward, terminated, truncated, _ = env.step(action) #与环境交互
            done = terminated or truncated
            ep_reward += reward
            mask = 0.0 if done else 1.0
            next_option = option

            should_terminate = torch.bernoulli(term_prob).bool()
            if should_terminate and not done:
                next_option = agent.choose_option(next_obs, epsilon=epsilon)

            buffer.store(
                obs=torch.tensor(obs, dtype=torch.float32).to(args.device),
                action=torch.tensor(action).to(args.device),
                logp=logp.detach().to(args.device),
                reward=torch.tensor(reward, dtype=torch.float32).to(args.device),
                value=value.detach().to(args.device),
                done=torch.tensor(done).to(args.device),
                option=torch.tensor(option).to(args.device),
                terminate=torch.tensor(option != next_option).to(args.device),
                mask=torch.tensor(mask).to(args.device),
                hidden=h.squeeze(0).to(args.device),
                term_prob=term_prob.detach().to(args.device),
            )

            obs = next_obs
            option = next_option
            total_steps += 1

            if done:
                writer.add_scalar("train/episode_return", ep_reward,  episode)
                ep_return_list.append(ep_reward)
                ep_reward = 0
                episode += 1
                obs, _ = env.reset()
                option = agent.choose_option(obs, epsilon=epsilon)

        agent.update(buffer, obs, option)

        if agent.update_cnt % args.eval_interval == 0:
            agent.evaluate(total_steps)

    writer.close()

if __name__ == "__main__":
    train()
```
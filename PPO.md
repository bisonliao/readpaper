**Proximal Policy Optimization Algorithms**

### Introduction

PPO方法希望解决：

1. RL方法是可扩展的（支持大模型、可以并行的训练）
2. 样本数据利用率高
3. 健壮稳定（能成功处理各种任务而不需要调超参数）

论文把PPO与TRPO做对比：PPO和TRPO一样具备样本的高利用率和可靠的性能，同时又不会像TRPO那样复杂的需要二阶导。



### Algorithm

#### 推导

下图是如何从普通的策略梯度，推出PPO的梯度：

![image-20250428170541008](img/image-20250428170541008.png)

#### 算法伪代码：

![image-20250428172930277](img/image-20250428172930277.png)

在 **PPO 的 Algorithm 1** 中，使用 **N 个并行 Actor（1 到 N）** 的主要目的是 **提高数据收集效率**，从而加速训练。但如果只想用 **1 个 Actor**，仍然可以训练，只需调整部分超参数。下面详细解释：

**并行 Actor 的作用**：

1. 加速数据收集：
   - 多个 Actor 同时与环境交互，能在相同时间内收集更多样本。
   - 例如，N=8，每个 Actor 跑 T=2048 步 → 总数据量 = 8×2048=16,384 步/迭代。
2. 降低样本相关性：
   - 不同 Actor 在不同环境实例（或不同随机种子）中运行，数据更具多样性。
3. 适用于分布式训练：
   - 在 CPU/GPU 集群上，多个 Actor 可以并行执行，提高硬件利用率。

这N个Actor会在每个iterator都对齐参数。



上面的伪代码也是醉了，让AI帮我生成了一版：

```python
Initialize trained_policy with random parameters θ
Initialize value_function with parameters φ

for iteration in range(max_iterations):
    
    # 1. 收集数据，使用 old_policy（即 θ_old = θ）
    # 本质上还是用当前策略与环境交互来收集数据，因为每个回合开始前，都会更新old_policy
    # 所以有的代码这里写的是当前策略与环境交互，是没有问题的。
    θ_old ← θ
    old_policy ← copy of trained_policy with frozen θ_old
    
    trajectories = []
    for actor in range(num_actors):           # 并行收集经验
        trajectory = collect_trajectory(old_policy, env, horizon)
        trajectories.append(trajectory)
    
    # 2. 处理数据，计算 advantage 和目标
    for trajectory in trajectories:
        for t in trajectory:
            # Generalized Advantage Estimation (GAE)
            δ_t = r_t + γ * V_φ(s_{t+1}) - V_φ(s_t)
            A_t = compute_GAE(δ_t, ...)
            R_t = A_t + V_φ(s_t)    # TD目标
    
    # 3. 用 fixed old_policy 和 current trained_policy 训练
    # old_policy在这里主要作用就是用来计算KL散度和重要性采样的比率
    for epoch in range(K_epochs):           # 多次遍历数据，但次数不能太多
        for minibatch in trajectories:
            
            # 当前策略的概率
            π_θ(a_t | s_t) = trained_policy.probability(s_t, a_t)
            # 旧策略的概率（不变）
            π_θ_old(a_t | s_t) = old_policy.probability(s_t, a_t)
            
            # 重要性采样比值
            r_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
            
            # PPO-Clip 损失
            L_clip = min(r_t * A_t, clip(r_t, 1 - ε, 1 + ε) * A_t)
            
            # Value函数损失
            L_value = (V_φ(s_t) - R_t)^2
            
            # 总损失（注意负号）
            loss = -mean(L_clip) + c1 * L_value - c2 * entropy(π_θ(. | s_t))

            # 更新策略参数θ 和价值函数参数φ
            update(trained_policy.parameters, ∇_θ loss)
            update(value_function.parameters, ∇_φ L_value)


```

#### PPO相比传统Actor-Critic方法的优势：

![image-20250520095951439](img/image-20250520095951439.png)

#### 环境初始化的问题：

PPO算法，收集一次固定步数的数据，就利用这些数据若干次；然后又收集新的数据。 我想知道假设是对于bipedalWalker这样的回合任务，两种情况那种好？还是没有区别？ 

1. 情况一：每次收集前都主动reset环境，开始一轮新的回合。收集过程中如果遇到done为True，就reset。
2.  情况二：开始训练的时候reset环境，后面一直按次序收集，不再主动reset，除非遇到done标记为True就reset一下

我从环境鲁棒性的角度，倾向于情况一，如果环境有bug，那么reset可以让环境恢复。但有的环境reset操作的开销很大。

但是情况二的优势是更利于探索，例如一个回合里就是要很多步之后才能到达目标位置。如果每回都reset，不利于深入后续状态。当然，可以要求每轮收集的步数尽量多一些也能弥补这个缺点。

#### 随机打散和分mini batch

在RPO的笔记中做实验发现，随机打散和mini batch对训练效果有明显影响，当然分minibatch会让训练时间变长。

结论是：要做随机打散和分mini batch，不要偷懒

另外要特别小心在用GAE或者MTC计算优势函数的时候，done字段的意义和last value的处理。

### Experiments

下图是在一系列任务上的与其他算法的性能对比

![image-20250428172038764](img/image-20250428172038764.png)

### show me the code

#### 离散动作空间

我自己写的PPO代码有很多细节不到位，直接贴CleanRL的代码吧：

```python
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import ale_py #这个虽然是灰的，也不能删除
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 4000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"logs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    total_reward = 0

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            #统计环境1交互过程中的每个回合的total reward
            if terminations[0]:
                writer.add_scalar("steps/total_reward_of_env0", total_reward, global_step)
                total_reward = 0
            else:
                total_reward += reward[0]

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                print(infos["final_info"])
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("steps/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("steps/episodic_length", info["episode"]["l"], global_step)



        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch, remove n_envs dim
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("steps/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("steps/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
```

上面的代码，在轨迹收集和保存方面，有很多细节考究：

1. 固定收集n_steps个时间步，这些时间步通常是跨回合的。如果是n_envs个并发环境，那么每个环境都固定收集n_steps个时间步
2. obs / action / logprob / reward / value / done六个字段，必须是指代同一个时刻的：
   1. obs：t 时刻的观测
   2. action / logprob：t 时刻，策略网络针对obs做出的动作和动作概率log
   3. reward：对环境施加action后，获得的奖励
   4. value：t 时刻的观测 obs 对应的价值评估
   5. done：t 时刻的obs是不是一个终止状态。这个字段很容易出问题，step函数返回的done，已经是 t+1 时刻的done了。
3. 固定收集的n_steps个时间步，保存在数组里，相同下标的不同元素有对应关系：
   1. 如果是只有一个单发环境，就保存在二维数组里，维度分别是steps, field_name。
   2. 如果是多个并发环境，就保存在三维数组里。维度分别是 env, steps,  field_name。
4. 计算GAE，最后一个时间步（SARVD信息）依赖的下一个时间步的done和value信息不在buffer里，要额外计算一下放进去。（next_value, next_done）
5. 并发多个环境收集的时间步，一开始保持n_envs这个维度作为批量进行GAE计算、划分minibatch，然后在更新模型参数的时候，把n_envs和n_steps两个维度打平成一个维度了。



环境的reset也很需要考究：

1. 在开始收集前，reset一下，后面不再反复reset而打断回合，让回合一直走下去、收集下去
2. 上面的代码，因为使用了SB3的VecEnv，它会自动reset并继续新的回合，所以后面没有reset了。如果是gym典型的单环境，遇到一个episode结束或者截断，需要显式的reset，修改obs和done字段的。



还有一个问题就是：如果不是环境模拟器，是真实的游戏环境，在收集完num_steps个时间步后，程序就去做GAE计算/模型更新等操作，这个操作可能是很费时的，那这时候环境会不会发生不可控的变化呢？会导致：

- 状态错位：环境已经推进但 agent 不知道；
- 行为延迟：模型更新后的策略要等一轮后才能生效；
- **安全风险**（在机器人控制中）：可能因为控制滞后导致硬件损伤；
- GAE 计算所基于的轨迹与现实错位，性能显著下降。



#### 连续动作空间

能够很好的收敛：

![image-20250524162218274](img\image-20250524162218274.png)

代码还可以改进的地方：

优化网络参数前，做随机打散，并分多个mini batch进行update

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
        '''dist = Independent(Normal(mean, std), 1)
        action_raw = dist.rsample()'''
        action_raw = mean
        action = torch.tanh(action_raw) * 2.0
        action_np = action.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        state = next_state
        env.render()


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


**Hierarchical Deep Reinforcement Learning: integrating Temporal Abstraction and Intrinsic Motivation**

### Introduction

还是老生常谈的稀疏奖励带来的挑战问题。

我们提出了一个框架，分层组织的深度强化学习模块在不同的时间尺度上工作。模型在两个层次上做出决策

1. 顶层模块（元控制器）接受状态并选择一个新目标
2. 低层模块（控制器）使用状态和选择的目标来选择动作，直到达到目标或情节终止。然后元控制器选择另一个目标并重复这两个步骤。

我们在不同的时间尺度上使用随机梯度下降来训练我们的模型，以优化预期的未来内在奖励（控制器）和外在奖励（元控制器）。

我们的方法在两个典型的奖励大范围延迟的任务上表现突出：

1. 一个离散的随机决策过程，在这个决策过程里，在获得最佳的外部奖励之前，须经过一长串状态转换
2. 一个典型的ATARI 游戏：蒙特祖玛的复仇

### Literature Review

提到了Sutton提出的options框架。Options框架为强化学习中的时间抽象提供了严谨的数学基础，同时也启发了后续许多分层强化学习方法的发展，包括h-DQN在内。

论文中提到的多时间尺度，可以这样理解，多时间尺度抽象与人类决策机制高度吻合：

1. **战略层**：规划未来几周/月的目标(极慢时间尺度)
2. **战术层**：制定每日计划(中等时间尺度)
3. **执行层**：完成具体动作(快速时间尺度)

论文提出的方法，没有为每个options训练一个Q function网络，而是只训练一个Q function网络，options作为该网络的一个输入以应对多个options的需要。

这样做有两个好处：

1. 在不同的options间可以共享训练
2. 对于大量的options的场景，该方法有更好的扩展性

论文又扯了一下 内在激励RL（我理解例如 RND、ICM）、面向对象的RL、DRL、认知科学与神经科学 等相关概念和子领域。

### Model

#### 原理



![image-20250612161933274](img/image-20250612161933274.png)

![image-20250612171447688](img/image-20250612171447688.png)

#### 伪代码

![image-20250612172535564](img/image-20250612172535564.png)

#### 可以用于连续动作空间吗？

![image-20250612202946113](img/image-20250612202946113.png)

#### 如何推理

训练收敛后，推理不是只使用Q1网络，必须同时使用 Q2（选 subgoal）+ Q1（执行 subgoal）

### Experiments

![image-20250612174102939](img/image-20250612174102939.png)

第二个任务，蒙特祖玛的复仇，设计细节就很复杂，不摘抄了，贴一下结果：

![image-20250612174522856](img/image-20250612174522856.png)

### Conclusion

### bison的实验

#### frozen lake

[官方文档在这里](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)

代码如下：

```python
import copy
import datetime
import random
import time
from typing import SupportsFloat, Any
import numpy as np
import torch.nn as nn
import torch
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from collections import deque, namedtuple
from torch.optim import Adam
from torch.utils.tensorboard import  SummaryWriter

device="cpu"
writer = SummaryWriter(log_dir=f'./logs/hDQN_FrozenLake_{datetime.datetime.now().strftime("%m%d_%H%M%S")}')

class CustomFrozenLake(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.map_size = 8
        mapname = f'{self.map_size}x{self.map_size}'
        self.env = gym.make('FrozenLake-v1', desc=None, map_name=mapname, is_slippery=True, render_mode=render_mode)
        self.map = copy.deepcopy(self.env.unwrapped.desc) #type:np.ndarray
        print(self.map)

        self.map[0,0] = b'F'
        self.agent_pos = None

    def pos2xy(self, pos:int):
        row = pos // self.map_size
        col = pos - row * self.map_size
        return row, col

    def _add_agent_chn(self, state:np.ndarray):

        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)

        newchn = np.zeros_like(state, dtype=np.float32)
        row, col = self.pos2xy(self.agent_pos)
        newchn[0, row, col] = 1.0
        result =  np.concatenate([state, newchn], axis=0)
        return result
    def hasReachGoal(self):
        row, col = self.pos2xy(self.agent_pos)
        if row == self.map_size-1 and col == self.map_size-1:
            return True
        else:
            return False
    def hasReachSubgoal(self, subgoal:int):
        row, col = self.pos2xy(self.agent_pos)
        gr, gc = self.pos2xy(subgoal)
        if row == gr and col == gc:
            return True
        else:
            return False



    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.agent_pos = next_obs
        row, col = self.pos2xy(next_obs)
        next_state = copy.deepcopy(self.map)
        next_state[row, col] = b'A' # agent
        next_state = next_state.view(np.uint8) / 255.0
        next_state = self._add_agent_chn(next_state)
        return next_state, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.agent_pos = obs
        row, col = self.pos2xy(obs)
        state = copy.deepcopy(self.map)
        state[row, col] = b'A'  # agent
        state = state.view(np.uint8) / 255.0
        state = self._add_agent_chn(state)
        return state, info

    # 得到可以作为子目标的位置
    def get_valid_subgoal(self):
        subgoal = copy.deepcopy(self.map)
        return (subgoal != b'H').astype(np.int32)

def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

'''
输入 B x c x h x w形状的地图和 B x 2形状的subgoal
会给地图拼接一个单独的通道，用来表示subgoal的空间信息
经过各自的特征提取层后，拼接特征，然后经过全连接层输出每个动作的Q值
'''
class Q1Network(nn.Module):
    def __init__(self, c:int, h:int, w:int, subgoal_dim:int, action_dim:int):
        super().__init__()

        self.action_dim = action_dim

        self.map_feat = nn.Sequential(
            orthogonal_layer_init(nn.Conv2d(c+1, 16, kernel_size=3, padding=1)),
            nn.ReLU(),
            orthogonal_layer_init(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten()
        )

        # 测试获得输出featuremap的尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, c + 1, h, w)
            dummy_feat = self.map_feat(dummy)
            feature_dim1 = dummy_feat.shape[1]

        feature_dim2 = 32

        self.subgoal_feat = nn.Sequential(
            orthogonal_layer_init(nn.Linear(subgoal_dim, 32)),
            nn.ReLU(),
            orthogonal_layer_init(nn.Linear(32, feature_dim2)),
            nn.ReLU(),
        )


        self.out =  nn.Sequential(
            orthogonal_layer_init(nn.Linear(feature_dim1+feature_dim2, action_dim)),
        )

    def _add_subgoal_chn(self, map: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        """
        给输入 map 添加一个子目标通道，子目标由 subgoal 坐标指定。

        参数:
            map (Tensor): 输入地图张量，形状为 (B, C, H,W)
            subgoal (Tensor): 子目标坐标，形状为 (B, 2)，每行为 (y, x)

        返回:
            Tensor: 形状为 (B, C+1, H, W)，在末尾添加了子目标 mask 通道
        """
        B, C, H,W = map.shape
        assert subgoal.shape == (B, 2), f"Expected subgoal shape (B, 2), got {subgoal.shape}"

        # 初始化子目标 mask
        goal_mask = torch.zeros(B, 1, H,W, device=map.device, dtype=map.dtype)

        x = subgoal[:, 1]  # 列
        y = subgoal[:, 0]  # 行

        # 生成 batch 索引
        batch_idx = torch.arange(B, device=map.device)

        # 设置目标位置为 1（每个样本的目标位置）
        goal_mask[batch_idx, 0, x, y] = 1.0

        # 拼接通道：在 dim=1 上拼接
        map_with_goal = torch.cat([map, goal_mask], dim=1)

        return map_with_goal
    def forward(self, state:torch.Tensor, subgoal:torch.Tensor):

        x = self._add_subgoal_chn(state, subgoal)

        B, C, H,W = state.shape
        assert subgoal.shape == (B, 2), f"Expected subgoal shape (B, 2), got {subgoal.shape}"
        WH = max(H,W)
        feat1 = self.map_feat(x)
        feat2 = self.subgoal_feat(subgoal / WH)
        feat = torch.cat([feat1, feat2], dim=1)
        qvalue = self.out(feat)
        return qvalue
    def epsGreedy(self, state:torch.Tensor, subgoal:torch.Tensor, epsilon):
        if random.random() < epsilon:
            B, C, H, W = state.shape
            qvalue = torch.rand((B,self.action_dim), dtype=torch.float32, device=state.device)
        else:
            qvalue = self.forward(state, subgoal) #type:torch.Tensor
        return qvalue.argmax(dim=1)

'''
输入一个地图，其中一个通道包含了agent所在的位置信息，经过特征提取和转换，得到每个可能的位置的Q值
'''
class Q2Network(nn.Module):
    def __init__(self, c, h, w, valid_position_mask: np.ndarray, env:CustomFrozenLake):
        super().__init__()

        self.env = env

        self.map_feat = nn.Sequential(
            orthogonal_layer_init(nn.Conv2d(c, 16, kernel_size=3, padding=1)),
            nn.ReLU(),
            orthogonal_layer_init(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.valid_position_mask = torch.tensor(valid_position_mask, dtype=torch.bool)
        print(f'{self.valid_position_mask}')
        assert len(valid_position_mask.shape) == 2 and valid_position_mask.shape[0] == h and \
               valid_position_mask.shape[1] == w

        # 测试获得输出featuremap的尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy_feat = self.map_feat(dummy)
            feature_dim = dummy_feat.shape[1]


        self.out = nn.Sequential(
            orthogonal_layer_init(nn.Linear(feature_dim,  h*w)),
        )
    def forward(self, state:torch.Tensor):
        B, C, H, W = state.shape

        mask = self.valid_position_mask.unsqueeze(0).expand(B, -1, -1).to(device)

        x = self.map_feat(state)
        x = self.out(x) #type:torch.Tensor
        x = x.reshape((B,  H, W))
        neg_inf = float('-inf')
        x = torch.where(mask == 1, x, neg_inf)

        # agent所在当前位置不要选中作为suggoal
        row, col = self.env.pos2xy(self.env.agent_pos)
        x[:, row, col] = neg_inf

        return x #返回的是一个二维的对应地图形状的 Q值

    def epsGreedy(self, state:torch.Tensor, epsilon):
        B, C, H, W = state.shape
        if random.random() < epsilon:
            x = torch.rand((B,H,W), dtype=torch.float32, device=state.device)
            mask = self.valid_position_mask.unsqueeze(0).expand(B, -1, -1).to(device)
            neg_inf = float('-inf')
            x = torch.where(mask == 1, x, neg_inf)
            # agent所在当前位置不要选中作为suggoal
            row, col = self.env.pos2xy(self.env.agent_pos)
            x[:, row, col] = neg_inf
        else:
            x = self.forward(state)
        x = x.reshape((B, -1)) #统一展平,方便计算argmax
        pos = x.argmax(dim=1)
        return pos # shape:(B,)

# 经验回放缓冲区
Transition = namedtuple('Transition', ('input', 'output', 'reward', 'next_input', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = int(  (self.position + 1) % self.capacity  )

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Args:
    lr = 1e-4
    gamma = 0.999
    eps1_start = 1.0
    eps1_decay = 0.99
    eps1_end = 0.1

    eps2_start = 1.0
    eps2_decay = 0.99
    eps2_end = 0.1

    num_episodes = 10000

    buf_size = 1e6
    batch_sz = 64

    map_w = 8
    map_h = 8
    map_c = 2
    subgoal_dim = 2
    action_dim = 4

    update_target_network_interval = 4

class Critic:
    def __init__(self, env:CustomFrozenLake):
        self.env = env
    # todo:内部激励还是比较稀疏的，可能不太好...
    def getIntrinsicReward(self, subgoalInt):
        if self.env.agent_pos == subgoalInt:
            return 1
        else:
            return 0

class hDQNAgent:
    def __init__(self):
        self.env = CustomFrozenLake()
        valid_subgoal = self.env.get_valid_subgoal()
        self.q1 = Q1Network(Args.map_c,Args.map_h, Args.map_w,   Args.subgoal_dim, Args.action_dim).to(device)
        self.q2 = Q2Network(Args.map_c,Args.map_h, Args.map_w,   valid_subgoal, self.env).to(device)

        self.target_q1 = Q1Network(Args.map_c, Args.map_h, Args.map_w,   Args.subgoal_dim, Args.action_dim).to(device)
        self.target_q2 = Q2Network(Args.map_c, Args.map_h, Args.map_w,  valid_subgoal, self.env).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.optimizer1 = Adam(self.q1.parameters(), lr=Args.lr)
        self.optimizer2 = Adam(self.q2.parameters(), lr=Args.lr)

        self.D1 = ReplayBuffer(Args.buf_size)
        self.D2 = ReplayBuffer(Args.buf_size)

        self.epsilon2 = Args.eps2_start
        self.epsilon1_dict = dict()
        self.subgoal_record = dict() # 每个subgoal 对应一个deque(maxlen=100)，里面是成功还是失败的结果 1/0

        self.critic = Critic(self.env)
        self.episode = 0
        self.total_step = 0

    def decay_epsilon(self, episode):
        self.epsilon2 = max(Args.eps2_end, self.epsilon2*Args.eps2_decay)
        writer.add_scalar('episode/epsilon2', self.epsilon2, self.episode)
        keys = self.epsilon1_dict.keys()

        minV, minV_updated = 1.0, False
        maxV, maxV_updated = float('-inf'), False
        rates = []
        for g in keys:
            suc_rate = 0
            if self.subgoal_record.__contains__(g):
                record = self.subgoal_record[g] #type:deque
                suc_rate = record.count(1) / (len(record)+1e-8)
            else:
                self.subgoal_record[g] = deque(maxlen=100)
            rates.append(suc_rate)
            if suc_rate > 0.7:
                self.epsilon1_dict[g] = max(Args.eps1_end, self.epsilon1_dict[g] * Args.eps1_decay)
                if self.epsilon1_dict[g] > maxV:
                    maxV = self.epsilon1_dict[g]
                    maxV_updated = True
                if self.epsilon1_dict[g] < minV:
                    minV = self.epsilon1_dict[g]
                    minV_updated = True
        if minV_updated:
            writer.add_scalar('episode/epsilon1_min', minV, self.episode)
        if maxV_updated:
            writer.add_scalar('episode/epsilon1_max', maxV, self.episode)
        writer.add_scalar('episode/subgoal_suc_rate_mean', sum(rates) / (len(rates)+(1e-8)), self.episode)



    def get_epsilon1(self, subgoalInt):
        if self.epsilon1_dict.__contains__(subgoalInt):
            return self.epsilon1_dict[subgoalInt]
        else:
            self.epsilon1_dict[subgoalInt] = Args.eps1_start
            return Args.eps1_start

    def add_subgoal_result(self, reached, subgoalInt):
        if self.subgoal_record.__contains__(subgoalInt):
            record = self.subgoal_record[subgoalInt]  # type:deque
            record.append(1 if reached else 0)
        else:
            record = deque(maxlen=100)
            record.append(1 if reached else 0)
            self.subgoal_record[subgoalInt] = record

    def update_target_network(self):
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        writer.add_scalar('episode/update_target', 1, self.episode)


    def train(self):

        for i in range(Args.num_episodes):
            self.episode = i+1

            # state和stateTensorshi一对，只要修改state，就一定初始化stateTensor
            state, _ = self.env.reset()
            stateTensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # subgoal, subgoalTensor,subgoalInt是绑定的，修改subgoal一定要修改其他两个
            subgoal = self.q2.epsGreedy(stateTensor, self.epsilon2 ) # type:torch.Tensor  (1,)
            subgoalInt = subgoal.cpu().item()
            row,col = self.env.pos2xy(subgoalInt)
            subgoalTensor = torch.tensor([[row, col]], dtype=torch.int32, device=device) # shape: (1,2)

            #print(f'begin train sub goal {subgoalInt} from {self.env.agent_pos}')
            writer.add_scalar('steps/subgoal', subgoalInt, self.total_step)

            done = False
            while not done:
                F = 0
                s0 =  copy.deepcopy(stateTensor)
                subgoalReached = False
                while not (done or self.env.hasReachGoal() or self.env.hasReachSubgoal(subgoalInt)):
                    eps1 = self.get_epsilon1(subgoalInt)
                    a = self.q1.epsGreedy(stateTensor, subgoalTensor, eps1)
                    a = a.squeeze(0).cpu().item()
                    next_state, f, terminated, truncated, info = self.env.step(a)
                    self.total_step += 1
                    nextStateTensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                    done = terminated or truncated
                    r = self.critic.getIntrinsicReward(subgoalInt)

                    if self.env.hasReachSubgoal(subgoalInt):
                        subgoalReached = True
                        writer.add_scalar('steps/reach_subgoal', 1, self.total_step)
                    if self.env.hasReachGoal():
                        writer.add_scalar('steps/reach_goal', 1, self.total_step)

                    self.D1.push( (stateTensor.squeeze(0), subgoalTensor.squeeze(0)), a, r, (nextStateTensor.squeeze(0), subgoalTensor.squeeze(0)), done )

                    loss1 = self.update_q1()
                    loss2 = self.update_q2()
                    if self.total_step % 200 == 0:
                        writer.add_scalar('steps/loss1', loss1, self.total_step)
                        writer.add_scalar('steps/loss2', loss2, self.total_step)
                        print(f'log loss2={loss2} at {self.total_step}')

                    F += f
                    state = next_state
                    stateTensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                self.add_subgoal_result(subgoalReached, subgoalInt)

                assert self.env.map[subgoalTensor[0, 0].cpu().item(), subgoalTensor[0, 0].cpu().item()] != b'H', \
                    print(f'{subgoalTensor},{self.env.map[subgoalTensor[0, 0].cpu().item(), subgoalTensor[0, 0].cpu().item()]}')
                self.D2.push( s0.squeeze(0), subgoalTensor.squeeze(0), F, nextStateTensor.squeeze(0), done) #如果只是达到子目标，done是false，如果达到最终目标或者回合长度超时，done是true
                if not done:
                    subgoal = self.q2.epsGreedy(stateTensor, self.epsilon2)  # type:torch.Tensor
                    subgoalInt = subgoal.cpu().item()
                    row, col = self.env.pos2xy(subgoalInt)
                    subgoalTensor = torch.tensor([[row, col]], dtype=torch.int32, device=device)
                    writer.add_scalar('steps/subgoal', subgoalInt, self.total_step)
                    #print(f'begin train sub goal {subgoalInt} from {self.env.agent_pos}')

            self.decay_epsilon(episode=i)
            if (i+1) % Args.update_target_network_interval == 0:
                self.update_target_network()

    def update_q1(self):
        if len(self.D1) < Args.batch_sz:
            return 0
        batch = self.D1.sample(Args.batch_sz)
        inputs, actions, rewards, next_states, dones = zip(*batch)
        stateTensors, subgoalTensor = zip(*inputs)
        nextStateTensor, _ = zip(*next_states)

        stateTensors = torch.stack(stateTensors)
        subgoalTensor = torch.stack(subgoalTensor)
        nextStateTensor = torch.stack(nextStateTensor)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)

        # 计算当前 Q 值
        q_values = self.q1.forward(stateTensors, subgoalTensor).gather(1, actions)  # 从 Q(s, a) 选取执行的动作 Q 值

        # 计算目标 Q 值
        next_q_values = self.target_q1.forward(nextStateTensor, subgoalTensor).max(1, keepdim=True)[0]  # 选取 Q(s', a') 的最大值
        target_q_values = rewards + Args.gamma * next_q_values * (1 - dones)  # TD 目标

        # 计算损失
        loss = nn.functional.mse_loss(q_values, target_q_values.detach())
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()

        return loss.item()

    def update_q2(self):
        if len(self.D2) < Args.batch_sz:
            return 0

        W = Args.map_w

        batch = self.D2.sample(Args.batch_sz)
        stateTensors, subgoalTensors, rewards, nextStateTensors, dones = zip(*batch)
        stateTensors = torch.stack(stateTensors)
        subgoalTensors = torch.stack(subgoalTensors)
        nextStateTensors = torch.stack(nextStateTensors)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  # (batch_size,) -> (batch_size, 1)

        # 计算当前 Q 值
        q_values = self.q2.forward(stateTensors) #计算子目标
        q_values = q_values.reshape((Args.batch_sz,-1)) #展平
        row = subgoalTensors[:, 0]  # shape: (B,)
        col = subgoalTensors[:, 1]  # shape: (B,)
        indexTensor = row * W + col  # shape: (B,)
        indexTensor = indexTensor.to(torch.int64).unsqueeze(1)
        q_values = q_values.gather(1, indexTensor)  # 从 Q(s, a) 选取执行的动作 Q 值

        # 计算目标 Q 值
        next_q_values = self.target_q2.forward(nextStateTensors)
        next_q_values = next_q_values.reshape((Args.batch_sz,-1))
        next_q_values = next_q_values.max(1, keepdim=True)[0]  # 选取 Q(s', a') 的最大值
        target_q_values = rewards + Args.gamma * next_q_values * (1 - dones)  # TD 目标

        # 计算损失
        loss = nn.functional.mse_loss(q_values, target_q_values.detach())
        self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer2.step()

        return loss.item()

def main():
    agent = hDQNAgent()
    agent.train()


main()

```


**MasteringDiverseDomainsthroughWorldModels**

nature'2025, from Google Brain



论文宣称使用统一配置在超过 150 个任务上训练，并且在 Minecraft 中仅依靠像素、稀疏奖励和从零开始的探索，最终学会获得钻石。Dreamer v3 证明了 world-model RL 可以从一个任务专用的研究方法，进一步发展成跨视觉、控制、游戏、稀疏奖励环境都具有相当竞争力的通用算法。它把 model-based RL 从“在特定任务上有效”，推进到了“有希望成为通用学习框架”的层次。

### 一、与Dreamer算法的对比

Dreamer v3 没有改变 Dreamer 的核心思想：

```
真实经验
  → 学习世界模型
  → 在世界模型中想象未来
  → 用 imagined trajectories 训练 actor 和 critic
  → 回到真实环境收集新数据
```

它的主要贡献是：把 Dreamer v1 中较依赖任务类型和超参数的实现，改造成一个能够处理不同输入、动作、奖励尺度和任务难度的统一算法。

论文声称 v3 在超过 150 个任务上使用同一套配置，并且从像素和稀疏奖励开始在 Minecraft 中学会获得钻石。Dreamer v3.pdf

#### v1 和 v3 的整体区别

| 方面            | Dreamer v1                              | Dreamer v3                                         |
| --------------- | --------------------------------------- | -------------------------------------------------- |
| 潜状态          | 连续高斯潜变量                          | 离散 categorical 潜变量                            |
| 世界模型        | 重建图像、预测奖励、KL 约束             | 加入 continuation 预测、双向 KL、free bits、unimix |
| critic          | 预测一个标量 value                      | 预测 return distribution                           |
| actor 更新      | 主要通过世界模型反向传播 value gradient | 使用 REINFORCE estimator，统一处理离散和连续动作   |
| 奖励/value 回归 | 普通标量回归                            | symlog、symexp two-hot                             |
| 探索控制        | 主要依赖动作噪声或 epsilon              | return normalization + entropy regularization      |
| 网络            | 普通 MLP、GRU 风格                      | Block GRU、RMSNorm、SiLU                           |
| 优化器          | Adam                                    | AGC + LaProp                                       |
| replay          | 普通 replay buffer                      | 更大 replay、online queue、存储和更新 latent state |
| 目标            | 主要验证 latent imagination             | 跨领域、跨奖励尺度、固定超参数的通用 RL            |

#### 1. 从连续潜状态改成离散潜状态

Dreamer v1 使用连续的高斯潜状态，例如：

```
z_t 是一个连续向量
```

Dreamer v3 把潜状态改成多个 categorical latent：

```
z_t = 多个离散变量
每个变量都有若干个可能的类别
```

直觉上，v3 不再让模型直接输出一个任意的连续向量，而是让它在多个离散类别之间选择概率。

例如某些 latent 维度可能隐式表示：

```
物体是否存在
机械臂处于哪种姿态
是否接触
当前处于哪个阶段
```

这些并不一定是人类可解释的标签，但离散表示更容易被序列模型预测，也便于快速进行 imagined rollout。

需要注意：v3 仍然使用随机采样，并通过 straight-through gradient 训练，不是把潜状态变成完全确定的符号。

#### 2. 世界模型增加了 continuation predictor

v1 主要预测：

```
图像
奖励
潜状态转移
```

v3 还预测：

```
当前 episode 是否继续
```

也就是加入了一个 continuation predictor：

```
c_t = 1：episode 还在继续
c_t = 0：episode 已经结束
```

这很重要，因为 imagined trajectory 可能提前终止。

例如在 Minecraft 中：

```
玩家死亡
```

之后就不应该继续把未来奖励累计进去。

因此 v3 的想象回报不仅考虑：

```
预测奖励
```

还会考虑：

```
当前状态是否还能继续产生未来奖励
```

#### 3. KL 约束从单一形式变成 KL balancing

v1 的世界模型大致有：

```
图像重建损失
+ 奖励预测损失
+ posterior 和 prior 之间的 KL 损失
```

v3 把 KL 相关目标拆成了两个方向。

第一项要求动力学预测器去匹配看过真实图像的表示：

```
prior 预测的 z_t
≈
encoder 看过真实 x_t 后得到的 z_t
```

第二项反过来要求表示模型学习出更容易被动力学模型预测的表示：

```
encoder 产生的 z_t
应该适合被 prior 预测
```

两项之间使用不同的 stop-gradient 和损失权重，这就是 KL balancing。

直觉上：

- 一部分梯度更新 transition model，让它更会预测；
- 另一部分梯度更新 representation，让它产生更容易预测的状态；
- 两者不能简单地用同一个 KL 梯度一起推，否则可能出现表示退化或动力学过度简化。

v3 还引入了 free bits：

```
KL 小于一定阈值时，不再继续惩罚
```

这样可以避免模型为了让 KL 很小，把潜状态压缩得过度，导致图像和任务信息丢失。

#### 4. 所有 categorical 分布加入 1% unimix

v3 将 categorical 分布设置成：

```
99% 神经网络输出
+ 1% 均匀分布
```

这样每个类别都保留极小的非零概率。

为什么需要这个？

如果某个类别概率变成严格的 0，那么：

```
log probability 可能变成负无穷
KL 散度可能突然爆炸
```

这会导致训练不稳定。

1% unimix 相当于给分布加了一个很小的概率地板，让：

```
概率不会严格变成 0
KL 不容易出现尖峰
```

#### 5. critic 从标量 value 改成 return distribution

v1 的 critic 大致是：

```
vψ(s_t) → 一个标量
```

v3 的 critic 则预测：

```
vψ(R_t | s_t) → return 的概率分布
```

也就是说，critic 不只说：

```
这个状态价值大约是 10
```

而是预测：

```
未来回报可能落在不同区间，各区间的概率是多少
```

最后可以再取期望得到普通 value。

这样做的原因是不同环境的回报可能差异巨大：

```
某个任务的 return 在 -1 到 1 之间
另一个任务的 return 可能在 0 到 100000 之间
```

如果所有任务都使用普通标量回归，梯度尺度很容易失控。

v3 使用按数量级展开的 categorical bins，让 critic 预测不同数量级的回报分布，从而提升跨任务稳定性。

#### 6. 奖励和价值预测改用 symlog 与 two-hot

v3 最核心的稳健性技巧之一，是不直接对原始奖励和回报做普通平方误差。

##### symlog

symlog 是一种对称的对数变换：

```
大数被压缩
小数基本保持不变
正负号保留
```

例如：

```
100000 和 100 的差别会被压缩
0.1 和 0.2 仍然可以被区分
-1000 仍然保持负号
```

这样可以避免大回报完全支配训练梯度。

##### symexp two-hot

对于奖励和 return，v3 让网络输出一个分布：

```
若干个按指数间隔排列的数值区间
```

目标值落在相邻两个 bin 之间时，不是只选择一个 bin，而是在两个相邻 bin 之间做 two-hot 插值。

这样比直接回归一个标量更稳定，尤其适合：

- 奖励尺度变化很大的环境；
- 回报分布有多个模式的环境；
- 存在极端 outlier 的环境；
- 稀疏奖励环境。

#### 7. actor 不再主要依赖 v1 的解析 value gradient

这是算法层面一个很重要的变化。

Dreamer v1 的核心特色是：

```
actor
  → 动作
  → 世界模型转移
  → 未来奖励和 value
```

然后把 value gradient 直接穿过世界模型反传回 actor。

Dreamer v3 则使用 REINFORCE estimator：

```
想象回报
− critic value
        ↓
作为 actor 的 advantage
        ↓
更新 log π(a_t | s_t)
```

也就是说，v3 仍然在世界模型中生成 imagined trajectories，但 actor 更新不再强依赖“穿过动力学模型的解析梯度”。

这样做的好处是：

- 同时适用于离散动作和连续动作；
- 不要求所有动作采样和动力学过程都适合路径导数；
- 更容易使用同一套算法覆盖 Atari、Minecraft、机器人控制等任务。

代价是 REINFORCE 本身可能方差更高，所以 v3 需要配合：

```
critic
+ return normalization
+ entropy regularization
```

来稳定训练。

#### 8. return normalization 改进跨任务探索

不同任务的奖励尺度和稀疏程度不同。

例如：

```
任务 A：每一步奖励 0.1
任务 B：几千步后偶尔奖励 1
任务 C：成功时奖励 10000
```

如果使用同一个 entropy coefficient：

```
奖励尺度大的任务可能几乎不探索
奖励尺度小的任务可能过度探索
```

v3 对 return 做基于百分位数的归一化：

```
使用第 5 到第 95 百分位的 return 范围
再用 EMA 平滑
```

但它不会简单地把所有小回报放大。

论文中特别强调：

- 大回报被适当缩放；
- 小于一定阈值的回报不继续放大；
- 防止稀疏奖励下的噪声被夸大；
- 同一个 entropy 系数可以在不同任务中工作。

因此 v3 的探索控制相比 v1 更系统。

#### 9. critic 加入 EMA 正则和 replay loss

v3 的 critic 训练相比 v1 更复杂。

首先，critic 的 return target 依赖 critic 自己的预测：

```
critic 预测 target
        ↓
用 target 再训练 critic
```

这可能造成自我追逐和不稳定。

v3 用 critic 参数的 exponential moving average 作为稳定参照，类似 target network，但不是完全复制 DQN 的实现。

其次，critic 不只从 imagined trajectories 学习，也从 replay trajectory 学习：

```
想象轨迹：主要 critic loss
真实 replay 轨迹：额外 critic loss
```

这样做可以让 critic 更充分利用已有经验，尤其在奖励难预测时更有帮助。

#### 10. 网络结构换成更适合扩展的版本

Dreamer v1 主要使用：

```
普通 MLP
ELU 激活
Adam
```

Dreamer v3 改成：

```
RMSNorm
SiLU
Block GRU
AGC
LaProp
```

其中 Block GRU 将循环权重分成多个块：

```
大循环状态
→ 分成多个 block
→ block 内计算
```

这样可以扩大记忆容量，同时避免普通大 GRU 的参数量和计算量快速增长。

AGC 则按照参数张量自身的规模进行梯度裁剪，而不是使用一个固定的全局阈值。这样在模型大小和损失尺度改变时，不需要重新调梯度裁剪参数。

#### 11. replay buffer 和训练调度也更工程化

v3 的 replay 设计包括：

- 更大的 replay capacity；
- online queue；
- uniform replay；
- 数据采集时存储 latent state；
- 训练后把更新过的 latent state 写回 replay；
- 用 replay ratio 控制每个环境步对应多少次梯度更新。

这让 v3 可以显式调节：

```
更多真实环境交互
vs
更多模型和策略更新
```

从而在计算资源允许时提高数据效率。

#### 12. 初始化奖励和价值输出为零

v3 还做了一个看似简单但很实用的修改：

```
reward predictor 的输出层权重初始化为 0
critic 的输出层权重初始化为 0
```

原因是随机初始化的 reward predictor 和 critic 可能一开始就预测出很大的虚假奖励或虚假价值。

这样 actor 可能被错误的 imagined reward 吸引，导致训练初期不稳定。

零初始化使模型初始时更接近：

```
没有强烈的虚假奖励信号
```

然后随着真实数据进入，奖励和价值逐步建立。

#### v3 仍然没有改变的部分

尽管做了大量修改，Dreamer v3 的核心骨架仍然是：

```
RSSM 世界模型
+ replayed experience
+ 潜空间 imagined rollout
+ actor-critic
+ λ-return
+ 真实环境持续收集经验
```

所以 v3 不是一个完全不同的算法，而是：

> 把 v1 的“可行的 latent imagination 算法”，改造成一个对奖励尺度、观察类型、动作类型、环境复杂度和模型规模都更稳健的通用版本。

#### 最值得记住的三类改动

如果只保留最重要的变化，可以记成三类。

**第一类：表示和世界模型更稳定**

```
连续 latent → 离散 latent
单一 KL → KL balancing + free bits
加入 continuation predictor
categorical distribution 加 unimix
```

**第二类：奖励和价值尺度更稳定**

```
标量回归 → symlog + symexp two-hot
标量 critic → distributional critic
普通 return 处理 → percentile return normalization
```

**第三类：跨任务工程稳健性更强**

```
Adam → AGC + LaProp
普通 GRU → Block GRU
普通网络 → RMSNorm + SiLU
固定简单 replay → 更大 replay + online queue + replay ratio
```

#### 最终效果和边界

v3 的目标不再只是：

```
在几个视觉控制任务上证明 latent imagination 可行
```

而是：

```
使用一套固定配置，跨不同领域直接运行
```

论文报告的领域包括：

- Atari；
- ProcGen；
- DMLab；
- Minecraft；
- BSuite；
- proprioceptive control；
- visual control。

它尤其展示了两点：

1. 在奖励尺度差异很大的任务之间，仍能使用统一配置；
2. 在 Minecraft 这种视觉、长时域、稀疏奖励环境中，不依赖人类示范和手工 curriculum，最终有训练实例获得钻石。

不过这不代表 v3 完全解决了探索问题。Minecraft 的结果仍然显示，单个 episode 中成功获得钻石的比例很低；它更准确的贡献是：

> 通过更稳健的表示、价值估计、探索正则和想象训练，让 Dreamer 在以前很难从零开始的任务上有机会持续发现并利用稀疏奖励。

### 二、代码

在[这里](https://github.com/danijar/dreamerv3)

[RoboBase](https://github.com/swirl-uk/robobase)也有实现该算法
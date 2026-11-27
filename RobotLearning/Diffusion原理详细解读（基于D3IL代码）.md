# Aligning DDPM：从专家动作到噪声监督的完整训练过程

本文对应 `.vscode/launch.json` 的 `D3IL: Aligning DDPM (0.25data, run.py)`，依据当前仓库代码整理。

先回答最关键的两个问题：

1. 每次计算 batch 的损失时，**每条样本独立随机抽取一个扩散阶段 `t`**，当前范围为 `0～23`。一个 batch 没有统一的 `t`。
2. 每条样本使用干净的专家动作和新采样的高斯噪声，**直接合成指定阶段的含噪动作**，再训练网络预测这份噪声。不需要先执行前一个或后一个阶段，也不需要先让模型生成训练输入。

## 1. 先解决“相邻去噪阶段是否形成链式依赖”的困惑

### 1.1 训练的输入和标签到底是什么

假设原始数据提供状态 `s` 和专家动作 `a`，标准化后的专家动作记为 `a_clean`。训练时构造：

```text
已知：状态 s，干净动作 a_clean
随机生成：阶段 t，标准高斯噪声 ε

含噪动作 a_noisy = A(t) × a_clean + B(t) × ε

模型输入：(a_noisy, t, s)
监督标签：ε
模型输出：ε_pred
损失：mean((ε_pred - ε)²)
```

`A(t)`、`B(t)` 是根据噪声调度预先计算好的系数，后面给出其定义。

这里不需要构造 `(a_noisy[k], a_noisy[k-1])` 这样的相邻状态训练对。当前配置 `predict_epsilon=true`，直接监督的是我们自己生成的 `ε`，而非相邻阶段的动作。因此，输入和标签都能在当前训练步内构造出来。

### 1.2 含噪动作为什么能直接生成

如果逐步进行前向加噪，确实存在递推关系。为避免代码从 0 开始的下标带来混淆，本小节暂用数学阶段 `k=1,...,24`，并令 `a^(0)=a_clean`：

$$
a^{(k)}=\sqrt{\alpha_k}\,a^{(k-1)}+\sqrt{\beta_k}\,z_k,
\qquad z_k\sim\mathcal N(0,I),\quad \alpha_k=1-\beta_k.
$$

但将递推式展开，例如前两步：

$$
a^{(1)}=\sqrt{\alpha_1}\,a_{clean}+\sqrt{\beta_1}\,z_1
$$

$$
a^{(2)}=\sqrt{\alpha_2\alpha_1}\,a_{clean}
+\sqrt{\alpha_2\beta_1}\,z_1+\sqrt{\beta_2}\,z_2.
$$

后两项是独立高斯随机变量的线性组合，其均值为 0，方差为：

$$
\alpha_2\beta_1+\beta_2
=\alpha_2(1-\alpha_1)+(1-\alpha_2)
=1-\alpha_1\alpha_2.
$$

因此，直接采样一个新的 `ε ~ N(0,I)` 就能得到相同的分布：

$$
a^{(2)}\overset{d}{=}
\sqrt{\alpha_1\alpha_2}\,a_{clean}
+\sqrt{1-\alpha_1\alpha_2}\,\epsilon.
$$

推广到任意阶段：

$$
a^{(k)}\overset{d}{=}
\sqrt{\bar\alpha_k}\,a_{clean}
+\sqrt{1-\bar\alpha_k}\,\epsilon,
\qquad \bar\alpha_k=\prod_{j=1}^{k}\alpha_j.
$$

`等分布` 的意思是：直接生成的样本，与逐步加噪到该阶段的样本服从相同的条件分布。它不要求两种方法得到同一条随机轨迹上的同一个数值。

所以，**递推过程有相邻依赖，但采样任意阶段的边缘分布不需要执行这条递推链**。代码中的 `q_sample()` 正是用这个展开后的公式直接采样。

这里的 `ε` 等价于前面多次加噪累积后、归一化成标准高斯的总噪声，并非递推式最后一步单独加入的 `z_k`。当前标签也是这个总噪声 `ε`。

### 1.3 对应到“第 k 次去噪的输入从哪里来”

如果你指的是从较高噪声阶段 `a^(k)` 去往较低噪声阶段 `a^(k-1)`：

- **训练时**：用上面的直接公式从 `a_clean` 构造 `a^(k)`，让网络预测其中的总噪声 `ε`。不需要先跑一次模型得到 `a^(k)`，也不用先构造 `a^(k-1)` 作为标签。
- **推理时**：只有状态、没有专家动作，才必须从随机噪声开始，执行 `a^(24) → a^(23) → ... → a^(0)`。此时下一步输入确实依赖上一步输出。

训练在多个 batch 中覆盖各个噪声阶段，同一个网络通过时间编码得知当前处于哪个阶段。训练完成后，再把这个网络按阶段串起来用于生成动作。

“第 5 次去噪”与“扩散阶段 `t=5`”也不是同一个概念。当前采样循环的索引顺序是 `23,22,...,0`，按调用次数数，第 5 次调用对应 `t=19`。

## 2. 当前调试入口实际使用的配置

配置由 [aligning_config.yaml](../configs/aligning_config.yaml)、[ddpm_agent.yaml](../configs/agents/ddpm_agent.yaml) 和 [launch.json](../.vscode/launch.json) 的覆盖参数共同决定。

| 配置 | 当前值 | 在训练中的含义 |
| --- | --- | --- |
| `agents` | `ddpm_agent` | 使用 `DiffusionAgent` |
| 外层模型 | `gc_diffusion.Diffusion` | 负责加噪、损失和采样 |
| 内层网络 | `DiffusionMLPNetwork` | 实际预测噪声的神经网络 |
| `train_batch_size` | `1024` | 每个完整 batch 包含 1024 个有效窗口 |
| `window_size` | `1` | 每个窗口只有一个轨迹时刻 |
| `obs_dim` / `action_dim` | `20` / `3` | 20 维状态、3 维位置增量 |
| `scale_data` | `true` | 状态和动作按训练集统计量标准化 |
| `n_timesteps` | `24` | `t` 的可选索引为 `0～23` |
| `t_dim` | `8` | 把每个样本的阶段索引编码为 8 维特征 |
| `beta_schedule` | `cosine` | 决定各阶段的加噪系数 |
| `predict_epsilon` | `true` | 用噪声作为监督标签 |
| `loss_type` | `l2` | 噪声预测的均方误差 |
| `goal_conditioned` | `false` | 不另外传入独立的 `goal` 张量；目标物体的信息已经包含在状态中 |
| 优化器 | Adam，`lr=1e-3` | 更新网络参数 |
| `seed` | `42` | 此调试入口未覆盖默认随机种子 |

`simulation.*` 三个参数控制训练结束后的仿真评估，不参与这里的训练样本构造。`D3IL_DISABLE_TENSORBOARD=1` 关闭日志上报，不改变损失计算。

`group=aligning_ddpm_seeds_0.25data` 是分组名称。当前数据加载实现按 `train_data_path` 指向的文件清单读取数据，没有根据 `group` 字符串自动截取 25% 数据；具体使用哪些示范取决于该清单的实际内容。

## 3. 原始监督数据：专家动作是怎么来的

位置：[Aligning_Dataset](../environments/dataset/aligning_dataset.py) 的 `__init__()`、`get_slices()`、`__getitem__()`。

这里使用轨迹时刻 `j` 表示机器人运动的时间，扩散索引仍记作 `t`，二者含义不同。

一个状态由以下信息按顺序拼接，合计 20 维：

```text
机器人期望末端位置 des_c_pos[j]     3 维
机器人实际末端位置 c_pos[j]         3 维
被推动物体位置                      3 维
被推动物体四元数                    4 维
目标物体位置                        3 维
目标物体四元数                      4 维
```

对应的 3 维专家动作是相邻时刻的期望位置之差：

```python
vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
# action[j] = des_c_pos[j + 1] - des_c_pos[j]
```

虽然变量名叫 `vel_state`，这里没有除以时间间隔，实际是位置增量，不是绝对位置或四元数动作，也不是严格意义上的速度。

`input_state[:-1]` 与这些位置增量逐项配对。`window_size=1` 时，单个数据集样本的形状为：

```text
state:  [1, 20]
action: [1, 3]
mask:   [1]
```

`get_slices()` 只在有效轨迹范围内生成窗口。DataLoader 打乱这些窗口并组成 batch，因此一个 batch 不必来自同一条示范，也不必包含相邻时刻。

完整 batch 的形状为 `[1024,1,20]`、`[1024,1,3]`、`[1024,1]`。DataLoader 没有设置 `drop_last=True`，最后一个 batch 可能更小。后文统一用 `B` 表示实际 batch 大小。

当前 `train_agent()` 虽然解包 `mask`，但不把它传入损失；此处有效窗口的筛选已经排除了轨迹末尾的补零区域。

## 4. 一次训练更新的实际调用链

```text
run.py: main()
  └─ DiffusionAgent.train_agent()
       └─ 从 train_dataloader 取得 state, action, mask
       └─ DiffusionAgent.train_step(state, action)
            ├─ 标准化 state、action
            ├─ gc_diffusion.Diffusion.loss(action, state, goal=None)
            │    ├─ 为 B 条样本分别采样 t
            │    └─ p_losses(x_start, state, goal, t)
            │         ├─ 采样 noise
            │         ├─ q_sample()：直接合成 x_noisy
            │         ├─ DiffusionMLPNetwork.forward()：预测 noise
            │         └─ 预测 noise 与真实 noise 计算 MSE
            ├─ zero_grad → backward → optimizer.step
            └─ 更新 EMA 参数
```

注意两层 `model`：`DiffusionAgent.self.model` 是外层 `Diffusion`；而 `gc_diffusion.py` 内的 `self.model` 是内层 `DiffusionMLPNetwork`。

训练调用的是外层 `.loss()`，不会走外层 `.forward()`。外层 `.forward()` 调用的是推理用的 `.sample()`。

### 第一步：标准化状态和专家动作

位置：[BaseAgent](../agents/base_agent.py) 初始化 scaler；[train_step()](../agents/ddpm_agent.py) 调用 [Scaler](../agents/utils/scaler.py)。

```python
state = self.scaler.scale_input(state)
action = self.scaler.scale_output(action)
```

标准化使用整个训练集有效状态、动作的逐维均值和标准差，而不是当前 batch 自己的均值和标准差：

```text
state_scaled  = (state_raw - state_mean) / (state_std + 1e-12)
action_scaled = (action_raw - action_mean) / (action_std + 1e-12)
```

所以进入 `Diffusion.loss()` 的 `x`，以及进入 `p_losses()` 的 `x_start`，都是**标准化后的干净专家位置增量**，形状 `[B,1,3]`。它不是状态，也不是已经带噪的动作。

### 第二步：为每条样本随机选择扩散索引 t

位置：[gc_diffusion.py](../agents/models/diffusion/gc_diffusion.py) 的 `loss()`，当前第 335 行。

```python
batch_size = len(x)
t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
```

当前 `self.n_timesteps=24`，所以上界 24 不包含在取值中。`t.shape == [B]`，例如一个四条样本的示意 batch 可以得到：

```text
样本             0    1    2    3
t                0    5   12   23
```

每条样本独立、均匀、有放回地抽取一个索引：可以重复，也不保证当前 batch 覆盖全部 24 个索引。同一条专家样本在下一次被取到时，会重新抽取 `t` 和噪声。

`t` 不是 epoch、batch 序号、机器人轨迹时刻，也不是按 batch 依次从 0 增长到 23。随机种子使随机数序列在相应可复现条件下可重现，不会让所有 batch 使用同一个 `t`。

当前窗口长度为 1，因此每条样本的 3 个动作坐标共用同一个 `t`，但每个坐标的噪声值分别采样。

### 第三步：生成作为监督标签的标准高斯噪声

位置：`p_losses()` 当前第 315 行。

```python
noise = torch.randn_like(x_start)
```

`noise` 与 `x_start` 形状相同，都是 `[B,1,3]`。每个元素采样自标准正态分布，记作 `ε ~ N(0,I)`。

标签不需要人工标注，因为本次噪声就是程序自己生成的。专家数据提供干净动作，程序据此在线构造新的噪声监督任务。

### 第四步：直接合成指定噪声阶段的动作

位置：`p_losses()` 第 317 行调用 `q_sample()`。

```python
x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
```

初始化时已经计算长度为 24 的固定系数数组：

```python
betas = cosine_beta_schedule(24)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
```

这些系数由调度决定，不是模型学习出来的。代码从 0 开始索引，记 `bar_alpha[t] = ∏(1-beta[j])`，乘积范围为 `j=0,...,t`，则：

```python
x_noisy = (
    extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
)
```

即 `A(t)=sqrt(bar_alpha[t])`、`B(t)=sqrt(1-bar_alpha[t])`。

`extract()` 位于 [diffusion/utils.py](../agents/models/diffusion/utils.py)：为每条样本取出其 `t` 对应的系数，并将 `[B]` 变为 `[B,1,1]`，广播乘到 `[B,1,3]` 上。

`noise` 显式传入 `q_sample()`，所以加噪使用的噪声和监督标签是同一个张量；不会在这里再抽一份不同的噪声。

**下标注意：代码的 `t=0` 已经表示一次前向加噪，不是完全干净。** 它对应第 1.2 节数学编号的 `k=1`，一般关系为 `k=t+1`。函数参数名 `x_start` 则始终表示干净动作。

按仓库实际 cosine 调度计算，系数约为：

| 代码 t | A(t)：干净动作系数 | B(t)：噪声系数 |
| --- | --- | --- |
| 0 | 0.997084 | 0.076314 |
| 5 | 0.920333 | 0.391137 |
| 12 | 0.655090 | 0.755551 |
| 23 | 0.002052 | 0.999998 |

例如设某条样本的标准化干净动作为 `[0.2,-0.1,0.3]`，本次随机噪声为 `[0.5,-1.0,0.25]`，抽到 `t=5`：

```text
x_noisy = 0.920333 × [0.2, -0.1, 0.3]
        + 0.391137 × [0.5, -1.0, 0.25]
        ≈ [0.379635, -0.483170, 0.373884]

输入：(这个 x_noisy, t=5, 标准化状态)
标签：[0.5, -1.0, 0.25]
```

这条含噪动作由干净动作直接算出，完全没有调用另一个去噪阶段。示例数值用于解释构造过程，不是一次实际训练 batch 的抓取结果。

### 第五步：内层网络前向预测噪声

位置：`p_losses()` 第 319 行；[DiffusionMLPNetwork.forward()](../agents/models/diffusion/diffusion_models.py)。

```python
x_recon = self.model(x_noisy, t, state, goal)
```

当前 `goal=None`，状态不进行扩散加噪。网络内部执行：

```text
t: [B]
  → SinusoidalPosEmb(8): [B,8]
  → Linear(8,16) → Mish → Linear(16,8): [B,8]
  → 增加长度为 1 的窗口维度: [B,1,8]

按最后一维拼接：
  x_noisy [B,1,3] + t_embedding [B,1,8] + state [B,1,20]
  → 网络输入 [B,1,31]

ResidualMLPNetwork：
  Linear(31,128)
  → 3 个双层残差块（num_hidden_layers=6）
  → Linear(128,3)
  → 噪声预测 [B,1,3]
```

这里的 `x_recon` 在 `predict_epsilon=true` 时实际上是 `epsilon_pred`，并没有在这一行重建干净动作。`t_dim=8` 指 8 维时间特征，和扩散步数 24、动作维度 3 是不同参数。

### 第六步：将预测噪声与生成的真实噪声计算 MSE

位置：`p_losses()` 第 323～326 行；[WeightedLoss / WeightedL2](../agents/models/diffusion/utils.py)。

```python
if self.predict_epsilon:
    loss = self.loss_fn(x_recon, noise, weights)
else:
    loss = self.loss_fn(x_recon, x_start, weights)
```

当前走第一个分支。`loss_type=l2` 逐元素计算平方误差，`weights` 沿调用链保持默认值 `1.0`，最后对全部元素取平均：

$$
L=\frac{1}{3B}\sum_{i=1}^{B}\sum_{d=1}^{3}
\left(\epsilon_{\theta,i,d}-\epsilon_{i,d}\right)^2.
$$

窗口维度为 1，所以平均分母为 `3B`。配置中的 `discount=0.99` 没有在当前这条训练调用链中作为 loss 权重使用。

此处没有先将预测噪声换算成动作、再与专家动作计算 MSE；没有调用 `p_sample()`、`q_posterior()` 或完整采样链。

`p_losses()` 的英文 docstring 描述了“重建初始样本并与初始样本比较”，这与当前 `predict_epsilon=true` 分支不一致。调试时应以实际条件分支为准；`false` 分支才直接监督干净动作。

### 第七步：反向传播和参数更新

位置：[DiffusionAgent.train_step()](../agents/ddpm_agent.py)。

```python
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
self.steps += 1
```

梯度更新包括时间编码中的线性层和残差 MLP。`t`、高斯噪声、专家数据及加噪调度系数不是要学习的参数。

当前每个训练步还会更新 EMA 参数（`update_ema_every_n_steps=1`、`decay=0.995`）。EMA 不改变本次噪声标签的构造。

## 5. 对照实际实现的一次训练伪代码

以下省略日志和 EMA，只保留当前配置实际执行的计算；`self` 表示外层 `DiffusionAgent`：

```python
# state_raw: [B,1,20]，action_raw: [B,1,3]
self.model.train()
state = self.scaler.scale_input(state_raw)
x_start = self.scaler.scale_output(action_raw)
diffusion = self.model

# 每条样本抽一个阶段，每个动作坐标抽一个标准高斯噪声。
B = x_start.shape[0]
t = torch.randint(0, 24, (B,), device=x_start.device).long()
noise = torch.randn_like(x_start)

# 按 t 取系数，直接构造含噪动作，无须递推生成中间状态。
coef_clean = diffusion.sqrt_alphas_cumprod[t].reshape(B, 1, 1)
coef_noise = diffusion.sqrt_one_minus_alphas_cumprod[t].reshape(B, 1, 1)
x_noisy = coef_clean * x_start + coef_noise * noise

# 一次网络前向；监督标签是本次加噪所用的同一个 noise。
noise_pred = diffusion.model(x_noisy, t, state, None)
loss = ((noise_pred - noise) ** 2).mean()

self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

## 6. 训练后的模型如何执行 24 步反向扩散并生成动作

本章分成两部分：第 6.1～6.5 节先不涉及仓库代码，只解释 DDPM 推理的原理；第 6.6 节以后再把每个数学步骤映射到当前实现。

### 6.1 推理要解决什么问题

训练数据中有干净专家动作，所以训练时可以把干净动作加噪，得到任意噪声阶段的训练输入。但推理时只有当前环境状态 `s`，没有专家动作。模型需要根据状态，从随机噪声中生成一个合理动作。

在 Aligning 机器人控制场景中，统一使用以下数学记号：

| 变量 | 维度 | 机器人控制中的含义 |
| --- | --- | --- |
| `s` | 20 维 | 当前条件状态：期望末端位置、实际末端位置、被推动物体位姿、目标物体位姿 |
| `x₀` | 3 维 | 最终希望生成的干净动作，即末端执行器的 XYZ 位置增量 `ΔXYZ` |
| `xₖ` | 3 维 | 扩散阶段 `k` 的含噪候选动作；它仍在动作空间中，但暂时不能发送给机器人 |
| `x₂₄` | 3 维 | 噪声最强的候选动作，推理时直接从标准高斯分布采样 |
| `ε` | 3 维 | 加在三个动作坐标上的标准高斯噪声 |
| `εθ(xₖ,k,s)` | 3 维 | 网络根据含噪动作、噪声阶段和机器人状态预测出的累计噪声 |

需要特别区分两类输入：

- `s` 是**条件状态**，告诉策略“机器人和物体现在处于什么情况”。它在一次 24 步生成过程中保持不变，也不参与扩散加噪。
- `xₖ` 是**动作生成变量**，表示“当前这一步正在加工的候选动作”。24 步扩散只作用于它。

因此，网络在阶段 `k` 的输入和输出可以概括为：

```text
输入：机器人/环境状态 s + 当前含噪动作 xₖ + 噪声阶段 k
输出：当前含噪动作中的 3 维噪声估计 ε̂ₖ
```

推理过程的目标是：

```text
x₂₄（随机高斯噪声）
  → x₂₃
  → x₂₂
  → ...
  → x₁
  → x₀（生成的干净动作）
```

因此，“24 次扩散”在推理阶段更准确地说是 **24 次反向扩散或去噪采样**。它不是让机器人运动 24 次，也不是对 20 维状态加噪，而是在内部构造一条 3 维候选动作的随机生成链。

### 6.2 为什么推理必须链式执行，而训练不需要

训练时，数据集给出了当前机器人状态 `s` 和对应的专家动作 `x₀=ΔXYZ`。因为已知干净专家动作，所以可以用闭式公式直接生成任意含噪动作 `xₖ`：

$$
x_k=\sqrt{\bar\alpha_k}\,x_0
+\sqrt{1-\bar\alpha_k}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

所以训练一个 batch 时，可以直接构造监督样本 `(s, xₖ, k) → ε`，不需要先生成 `x₁`，再生成 `x₂`，一直走到 `xₖ`。这里 `s` 只作为条件输入，公式右侧只有动作 `x₀` 被加噪。

推理时，策略能读到机器人状态 `s`，但不知道此时应执行的专家动作 `x₀`，因此无法使用上式直接得到某个中间动作。它只能先采样一个 3 维随机动作 `x₂₄ ~ N(0,I)`，在状态 `s` 的条件下生成 `x₂₃`，再把 `x₂₃` 作为下一步动作输入生成 `x₂₂`，依次走到 `x₀`。

换句话说：

- 训练学习的是：“给定机器人状态 `s`，如何识别含噪动作 `xₖ` 中的噪声”。
- 推理做的是：“固定机器人状态 `s`，把这个噪声识别能力在 24 个等级上依次使用，将随机动作变成符合当前场景的动作”。

24 步复用的是**同一个网络和同一组参数**，不是训练 24 个不同模型。网络通过阶段 `k` 的时间编码判断当前应处理多强的噪声。

### 6.3 一步反向扩散的完整原理

假设机器人当前状态为 `s`，反向扩散位于阶段 `k`，已有 3 维含噪候选动作 `xₖ`，现在要生成噪声更少的候选动作 `xₖ₋₁`。这一步只更新动作变量，不改变机器人状态 `s`，也不会真的驱动机器人。一步分为四个逻辑步骤。

#### 6.3.1 预测当前含噪动作中的总噪声

把当前 3 维含噪动作 `xₖ`、阶段 `k` 和 20 维机器人状态 `s` 输入训练好的网络：

$$
\hat\epsilon_k=\epsilon_\theta(x_k,k,s).
$$

网络输出 `ε̂ₖ`，形状与动作相同，也是 3 维。它预测的是从干净动作 `x₀` 直接构造 `xₖ` 时所对应的累计噪声，不只是前向扩散最后一步单独加入的那一小份噪声。

#### 6.3.2 由预测噪声反推出干净动作估计

将前向加噪公式移项，可以得到：

$$
\hat x_0^{(k)}=
\frac{x_k-\sqrt{1-\bar\alpha_k}\,\hat\epsilon_k}
{\sqrt{\bar\alpha_k}}.
$$

上标 `(k)` 表示这是在当前阶段 `k` 得到的 `x₀` 估计。每个阶段都会重新估计一次，后续不会一直沿用第一次的估计。

这个结果是“如果现在就猜测机器人最终应该执行什么 `ΔXYZ`”得到的 3 维动作估计。除最后一步外，算法不会把它直接发送给机器人，而是用它构造下一个候选动作 `xₖ₋₁` 的分布。

#### 6.3.3 计算下一阶段的条件分布

已知当前含噪候选动作 `xₖ` 和估计的干净动作 `x̂₀` 后，在机器人状态 `s` 的条件下，下一个候选动作服从高斯分布：

$$
p_\theta(x_{k-1}\mid x_k,s)
=\mathcal N\left(\mu_\theta(x_k,k,s),\tilde\beta_k I\right).
$$

先定义：

$$
\alpha_k=1-\beta_k,
\qquad
\bar\alpha_k=\prod_{j=1}^{k}\alpha_j,
\qquad
\bar\alpha_0=1.
$$

均值和方差为：

$$
\mu_\theta=
\frac{\beta_k\sqrt{\bar\alpha_{k-1}}}{1-\bar\alpha_k}\,\hat x_0^{(k)}
+\frac{(1-\bar\alpha_{k-1})\sqrt{\alpha_k}}{1-\bar\alpha_k}\,x_k,
$$

$$
\tilde\beta_k=
\frac{\beta_k(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}.
$$

直观上，均值综合了两部分信息：模型认为机器人最终应执行的动作，以及当前仍然含噪的候选动作。噪声调度决定两者在不同阶段的权重和下一步动作的不确定性。

#### 6.3.4 从这个分布采样下一阶段动作

当 `k>1` 时，采样：

$$
x_{k-1}=\mu_\theta+\sqrt{\tilde\beta_k}\,z,
\qquad z\sim\mathcal N(0,I).
$$

这里新采样的 `z` 也是 3 维，它与网络预测的 `ε̂ₖ` 不是同一个量：

- `ε̂ₖ` 用来估计当前 `xₖ` 中包含的累计噪声，进而估计 `x₀`。
- `z` 用来从 `xₖ₋₁` 的条件分布中抽取一个具体样本。

去噪过程中仍加入一小份随机噪声，并不矛盾。反向扩散是在当前机器人状态下，从动作概率分布中采样，而不是做确定性的“当前动作减去噪声”。这也使同一状态可能生成不同但都合理的推推动作。随着 `k` 下降，采样分布总体逐渐靠近干净动作分布；单条随机轨迹不要求 XYZ 每一维都单调变化。

### 6.4 24 步是怎样串起来的

完整生成过程可以写成：

```text
1. 读取并固定当前 20 维机器人/物体状态 s。
2. 随机采样一个 3 维初始候选动作 x₂₄ ~ N(0, I)。
3. 对 k = 24, 23, ..., 2：
   a. 输入状态 s、含噪动作 xₖ 和阶段 k，预测 3 维噪声 ε̂ₖ。
   b. 根据 ε̂ₖ 估计当前场景应执行的干净动作 x̂₀。
   c. 计算下一个候选动作 xₖ₋₁ 的均值和方差。
   d. 采样 3 维 xₖ₋₁，并把它作为下一轮的动作输入；状态 s 不变。
4. 对 k = 1：
   a. 再预测一次噪声并估计 x̂₀。
   b. 直接输出最终的 3 维动作 x₀，不再加入随机噪声。
```

链式依赖在这里真实存在：

$$
x_{24}\rightarrow x_{23}\rightarrow\cdots\rightarrow x_1\rightarrow x_0.
$$

例如，第 5 次网络调用接收的动作输入来自第 4 次调用的输出，不是重新从专家动作或高斯分布独立构造。但第 5 次接收的状态输入仍然是同一个 `s`。训练时随机学习各阶段，推理时才把这些动作阶段按顺序连接起来。

最后一步 `k=1` 时，由于 `bar_alpha₀=1`，上述后验方差变为 0。因此最后一步输出均值即可，不再采样随机扰动。

### 6.5 24 步去噪与机器人控制步的关系

一次 24 步反向扩散只生成**一个** 3 维动作增量：

```text
同一个环境观测 s
  → 内部执行 24 次网络前向
  → 得到一个 ΔXYZ
  → 机器人环境执行一次动作
  → 环境返回新观测
  → 再启动一条新的 24 步生成链
```

所以不能把 24 个扩散阶段理解成机器人连续执行 24 次位移。中间的 `x₂₃...x₁` 都是标准化动作空间中的潜在含噪候选动作，不是机器人实际末端位置，也不是 20 维环境状态，更不会发送给环境。只有最终 `x₀` 被还原到真实动作尺度并转换成控制命令。

从这里开始再对照当前仓库的实际实现。

### 6.6 代码入口：加载模型并准备条件状态

推理的实际路径是：

```text
DiffusionAgent.predict(state)
  → 标准化状态
  → 外层 Diffusion.forward() → sample() → p_sample_loop()
      → 从标准高斯随机动作开始
      → 对 t=23,22,...,0 逐步执行 p_sample()
          → 内层网络预测总噪声
          → predict_start_from_noise() 估计干净动作并按边界裁剪
          → q_posterior() 根据当前含噪动作和干净动作估计，计算下一阶段的均值、方差
          → 采样下一阶段；t=0 时不再额外加入随机噪声
      → 裁剪最终标准化动作
  → inverse_scale_output()：还原为真实尺度的位置增量
```

推理时的每一步不是简单地做 `当前动作 - 预测噪声`，而是使用上述系数和后验采样公式。当前 `diffusion_x=false`，没有追加固定在 `t=0` 的额外去噪迭代。

原理符号与代码变量的对应关系如下。最容易混淆的是数学阶段从 `k=1` 开始，而数组下标从 `t=0` 开始，因此 `k=t+1`。

| 原理中的量 | 当前代码中的量 | 含义 |
| --- | --- | --- |
| `s` | `state` / `input_state` | 固定的机器人和环境条件状态 |
| `xₖ` | `x` | 当前含噪候选动作 |
| `k` | `t + 1` | 数学阶段与代码下标的关系 |
| `εθ` | 内层 `self.model(...)` | 噪声预测网络 |
| `ε̂ₖ` | `self.model(x, t, s, g)` 的返回值 | 预测的 3 维累计噪声 |
| `x̂₀` | `p_mean_variance()` 中的 `x_recon` | 当前阶段估计的干净动作 |
| `xₖ₋₁` | `p_sample()` 的返回值 | 传给下一轮的候选动作 |

[run.py](../run.py) 在训练结束后加载 `eval_best_ddpm.pth`，然后进入环境仿真。这个检查点按验证集噪声损失选择；当前配置下，保存模型时使用 EMA 权重。

每次环境需要一个新动作，就调用一次 `DiffusionAgent.predict(obs)`。当前非视觉 Aligning 的 `obs` 包括上一步的目标 XYZ 和环境状态，合计 20 维。代码首先执行：

```python
state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
state = self.scaler.scale_input(state)
```

`window_size=1` 时，推理没有额外堆叠窗口维度，因此与训练的形状有一点区别：

| 张量 | 训练时 | 当前单条环境观测推理时 |
| --- | --- | --- |
| 条件状态 | `[B,1,20]` | `[1,20]` |
| 当前含噪动作 | `[B,1,3]` | `[1,3]` |
| 阶段索引 | `[B]` | `[1]` |
| 时间编码 | `[B,1,8]` | `[1,8]` |
| 拼接后的网络输入 | `[B,1,31]` | `[1,31]` |
| 噪声预测 | `[B,1,3]` | `[1,3]` |

`DiffusionMLPNetwork.forward()` 对二维和三维状态分别处理，最后一维的特征含义和网络参数相同。

当前 `use_ema=true`，`predict()` 临时切换到 EMA 权重并设置 `eval()`；整个函数有 `@torch.no_grad()`。所以推理不计算训练损失，也不反向传播或更新参数。`diffusion_kde=false`，直接走普通动作采样分支。

### 6.7 代码如何初始化 x₂₄

`Diffusion.forward()` 调用 `sample()`，后者根据当前状态构造动作形状 `(1,3)`，进入 `p_sample_loop()`：

```python
x = torch.randn(shape, device=self.device)  # 当前 shape = (1, 3)
```

这三个数起初只是标准高斯噪声，还不是可直接执行的真实尺度位移。采样链始终在标准化动作空间中运行。

训练时最高噪声阶段 `t=23` 的噪声系数约为 `0.999998`，干净动作系数约为 `0.002052`，因此推理用标准高斯作为链的初始分布。

为明确阶段与下标，沿用前面统一的数学记号：`x₀` 为干净动作，`x₂₄` 为最强噪声阶段。生成过程是：

```text
初始噪声 x₂₄
   ── t=23 ──→ x₂₃
   ── t=22 ──→ x₂₂
               ...
   ── t=1  ──→ x₁
   ── t=0  ──→ x₀：生成的干净动作
```

每一次箭头包含一次噪声预测和一次采样更新；`t=0` 也要执行网络前向，并不表示循环已经结束。

### 6.8 一步数学过程如何映射到代码

输入是当前含噪动作 `x`、阶段索引 `t` 和条件状态 `s`。下面的 `bar_alpha[t]` 使用代码的零起始下标；定义：

```text
alpha[t] = 1 - beta[t]
bar_alpha[t] = alpha[0] × ... × alpha[t]
bar_alpha_prev[t] = t > 0 时的 bar_alpha[t-1]，t=0 时取 1
```

这些数组在模型初始化时已经按当前 24 步调度算好，各次动作生成都复用它们。

**第一步：内层网络预测当前样本中的总噪声。**

```python
epsilon_pred = self.model(x, t, s, goal)
```

当前 `goal=None`。同一个网络接收不同的 `x` 和 `t`，输出形状仍为 `[1,3]`。这不是读取训练时保存的噪声，也不是知道了真实噪声；它是模型根据当前输入做出的估计。

**第二步：根据预测噪声，估计干净动作。**

`predict_start_from_noise()` 执行本节开头的换算，也就是：

```python
a_clean_pred = x / sqrt(bar_alpha[t]) \
    - sqrt(1 / bar_alpha[t] - 1) * epsilon_pred
```

然后因为 `clip_denoised=true`，在标准化空间按训练集动作的逐维最小值、最大值裁剪这个估计：

```python
a_clean_pred.clamp_(self.min_action, self.max_action)
```

这一刻的 `a_clean_pred` 是本步对最终干净动作的估计，通常还不会直接返回给环境。

注意变量名差别：训练的 `p_losses()` 中 `x_recon` 指网络预测的噪声；推理的 `p_mean_variance()` 中，`x_recon` 已经过 `predict_start_from_noise()` 换算，指的是干净动作估计。

**第三步：用当前含噪动作和干净动作估计，计算下一阶段的均值、方差。**

`q_posterior()` 使用预先计算好的系数：

$$
c_1[t]=\frac{\beta[t]\sqrt{\bar\alpha_{prev}[t]}}{1-\bar\alpha[t]},
\qquad
c_2[t]=\frac{(1-\bar\alpha_{prev}[t])\sqrt{\alpha[t]}}{1-\bar\alpha[t]}.
$$

$$
\mu_\theta=c_1[t]\,\hat a_{clean}+c_2[t]\,x,
\qquad
\tilde\beta[t]=\frac{\beta[t](1-\bar\alpha_{prev}[t])}{1-\bar\alpha[t]}.
$$

代码中分别对应 `posterior_mean_coef1`、`posterior_mean_coef2`、`posterior_variance`。均值依赖模型预测；方差使用调度推导出的固定值，当前网络没有额外预测方差。

函数虽然叫 `q_posterior()`，推理时传入的干净动作是模型估计值，不是已知的专家动作。

**第四步：采样下一阶段动作，作为下一次循环的输入。**

`p_sample()` 新生成一份高斯噪声 `z`，然后执行：

$$
x_{next}=\mu_\theta+\mathbf 1_{t>0}\sqrt{\tilde\beta[t]}\,z,
\qquad z\sim\mathcal N(0,I).
$$

实际代码使用对数方差：

```python
noise = torch.randn_like(x)
nonzero_mask = (1 - (t == 0).float()).reshape(...)
x_next = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
```

其中 `model_log_variance = log(max(posterior_variance, 1e-20))`；指数操作把对数方差变成标准差。

这份新噪声 `z` 与网络刚预测的总噪声 `epsilon_pred` 含义不同。前者用于按下一阶段的分布采样，后者用于估计干净动作。不能用 `epsilon_pred` 替换这里新采样的 `z`。

“去噪时为什么还要加噪声”：这里执行的是随机的反向扩散采样，需要按均值和方差得到下一阶段样本。加入的是该阶段规定强度的随机扰动，不是重新把整个动作初始化为纯噪声。随着阶段下降，整体趋向生成的干净动作；单条随机轨迹的数值或误差不保证每一步都单调减小。

### 6.9 代码如何连续循环 24 次

`p_sample_loop()` 的实际循环是：

```python
for i in reversed(range(0, self.n_timesteps)):
    timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
    x = self.p_sample(x, timesteps, state, goal)
```

索引按顺序遍历：

```text
23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0
```

| 网络调用次数 | 代码索引 t | 当前输入 | 输出传给谁 |
| --- | --- | --- | --- |
| 第 1 次 | 23 | 初始化的 `x₂₄` | 得到 `x₂₃`，传给第 2 次调用 |
| 第 2 次 | 22 | 第 1 次生成的 `x₂₃` | 得到 `x₂₂`，传给第 3 次调用 |
| 第 3～22 次 | 21～2 | 上一步的生成结果 | 继续传给下一步 |
| 第 23 次 | 1 | `x₂` | 得到 `x₁`，传给第 24 次调用 |
| 第 24 次 | 0 | `x₁` | 得到最终的 `x₀` |

这里有四点与训练不同：

- `t` 按顺序递减，不随机抽取；同一步推理 batch 内的样本使用同一个 `t`。
- 含噪动作 `x` 随循环更新，确实存在“上一步输出 → 下一步输入”的链式依赖。
- 24 次复用同一个已训练网络，不是 24 个分别训练的模型，参数在循环中保持不变。
- 条件状态 `state` 在 24 步中保持不变，循环里没有调用 `env.step()`。

每一步都会重新估计一次干净动作，不是第一步估计完成后就固定这个估计，后面只重复减噪声。

### 6.10 代码的最后一步为何不加噪声

当 `t=0` 时，`bar_alpha_prev[0]=1`，所以理论上：

```text
posterior_variance[0] = 0
posterior_mean_coef1[0] = 1
posterior_mean_coef2[0] = 0
```

此时 `model_mean` 就是该步估计并裁剪后的干净动作（实际浮点计算可能有舍入误差）。同时 `nonzero_mask=0`，所以最后返回均值，不再叠加随机扰动。

代码仍然会调用一次 `torch.randn_like(x)`，但这个随机张量最终乘以 0，不影响最后一步结果。此前步骤仍是随机采样，因此即使给定相同观测，多次调用整个 `predict()` 也可能生成不同动作。

### 6.11 从最终采样结果到环境中的一次动作

24 步结束后，`sample()` 再裁剪一次最终标准化动作。`DiffusionAgent.predict()` 恢复临时替换前的参数，并对采样结果进行反标准化：

```python
delta_xyz = action_scaled * (action_std + 1e-12) + action_mean
```

返回给 [aligning_sim.py](../simulation/aligning_sim.py) 的是 NumPy 数组，当前形状为 `[1,3]`。仿真端执行：

```python
pred_action = agent.predict(obs)                 # 内部完成 24 步去噪
pred_action = pred_action[0] + obs[:3]           # 位置增量 → 绝对目标位置
pred_action = np.concatenate((pred_action, [0, 1, 0, 0]), axis=0)
obs, reward, done, info = env.step(pred_action)  # 此时才推进环境
```

这里 `obs[:3]` 首步为初始末端位置，后续为上一步的目标位置。固定四元数补齐姿态后形成 7 维位姿命令。

因此，**24 步扩散产生一条 3 维位置增量，不是执行 24 次机器人动作**。环境完成一次动作后，再获取新状态，下一次 `agent.predict()` 会重新采样初始高斯噪声并执行一条新的 24 步链。当前实现不会把上一条链的中间噪声动作直接当作下一条链的起点。

### 6.12 对照实际实现的完整采样伪代码

下面只展开单次采样链，省略外层 EMA 切换和数据类型转换。`diffusion` 表示外层 `gc_diffusion.Diffusion`，其参数已加载，`state` 为标准化后的 `[1,20]` 状态：

```python
diffusion.eval()
with torch.no_grad():
    x = torch.randn((1, 3), device=state.device)

    for i in range(23, -1, -1):
        t = torch.full((1,), i, device=state.device, dtype=torch.long)

        # 同一个噪声预测网络重复使用 24 次。
        epsilon_pred = diffusion.model(x, t, state, None)

        # 每次重新估计干净动作，并裁剪到训练集标准化动作边界。
        a_clean_pred = (
            diffusion.sqrt_recip_alphas_cumprod[i] * x
            - diffusion.sqrt_recipm1_alphas_cumprod[i] * epsilon_pred
        )
        a_clean_pred.clamp_(diffusion.min_action, diffusion.max_action)

        # 用当前含噪动作与干净动作估计构造下一阶段的分布。
        mean = (
            diffusion.posterior_mean_coef1[i] * a_clean_pred
            + diffusion.posterior_mean_coef2[i] * x
        )
        std = (0.5 * diffusion.posterior_log_variance_clipped[i]).exp()
        z = torch.randn_like(x)
        x = mean + (1.0 if i > 0 else 0.0) * std * z

    action_scaled = x.clamp(diffusion.min_action, diffusion.max_action)
    # 外层 agent 随后对 action_scaled 反标准化，得到真实尺度位置增量。
```

### 6.13 调试时怎样看到完整的 24 步

在 `p_sample_loop()` 第 180 行设置断点，连续观察 `i`、`x`、`state`。进入 `p_sample()` 后继续观察 `model_mean`、`model_log_variance`、`nonzero_mask`，进入 `p_mean_variance()` 可观察已经换算为干净动作估计的 `x_recon`。

底层循环支持保存整个链。若在调试环境中已有 `diffusion` 和标准化后的 `state`，可单独执行：

```python
with torch.no_grad():
    final, chain = diffusion.p_sample_loop(
        state, None, (state.shape[0], 3), return_diffusion=True
    )

# 当前单条观测时：
chain.shape  # torch.Size([1, 25, 3])
```

第 2 维长度是 25：`chain[:,0]` 保存初始噪声，`chain[:,1]` 保存 `t=23` 更新后的动作，依此类推，`chain[:,24]` 是 `t=0` 更新后的动作，并与 `final` 相同。中间状态仍在标准化空间，也没有保证全部处于动作边界内。

这是直接调用底层循环进行观察，没有经过 `sample()` 的最终裁剪或 agent 的反标准化。不要直接把 `return_diffusion=True` 传给当前 `sample()`：它拿到元组后仍会调用 `.clamp_()`，无法直接处理全链返回值。

这段观察代码会重新生成一条链并消耗随机数，不是读取已经完成的 `predict()` 的历史结果。

| 对比项 | 一次训练更新 | 一次动作生成 |
| --- | --- | --- |
| 是否有干净专家动作 | 有 | 无 |
| 含噪输入怎么来 | 专家动作加上程序生成的噪声 | 首步为高斯噪声，后续为上一步生成结果 |
| 阶段 t | 每条样本独立随机抽取 | 按 `23→0` 顺序执行，同一步的 batch 共用当前阶段 |
| 内层网络前向次数 | 每条样本一个阶段，整个 batch 一次调用 | 24 次调用 |
| 是否需要相邻阶段的输出 | 不需要 | 需要 |
| 结果 | 噪声 MSE，用于更新参数 | 生成的位置增量 |

训练期间还有 `DiffusionAgent.evaluate()`，它同样调用 `.loss()` 并重新采样 `t` 和噪声。因此代码里叫作 `test_mse` 的验证指标，在当前配置下也是噪声预测 MSE，而非完整生成动作与专家动作的 MSE。训练结束后的 `env_sim.test_agent()` 才进行环境仿真评估。

## 7. 断点处建议观察的变量

在 `gc_diffusion.py` 的 `p_losses()` 中，按顺序观察：

| 断点执行后 | 变量 | 当前入口的形状 | 应理解为什么 |
| --- | --- | --- | --- |
| 进入函数 | `x_start` | `[B,1,3]` | 标准化后的干净专家位置增量 |
| 进入函数 | `state` | `[B,1,20]` | 标准化后的条件状态 |
| 进入函数 | `t` | `[B]` | 每条样本随机抽取的噪声阶段索引 |
| 第 315 行 | `noise` | `[B,1,3]` | 本次采样的标准高斯噪声，也是监督标签 |
| 第 317 行 | `x_noisy` | `[B,1,3]` | 直接加噪生成的网络输入 |
| 第 319 行 | `x_recon` | `[B,1,3]` | 当前配置下实际为预测噪声 |
| 第 324 行 | `loss` | 标量 | 预测噪声与 `noise` 的均方误差 |

可以在调试控制台验证：

```python
t[:10]
(t.min().item(), t.max().item())
x_start.shape, noise.shape, x_noisy.shape, x_recon.shape
((x_recon - noise) ** 2).mean().item()
loss.item()
```

最后两个值应一致。不要把当前 `x_recon` 当作干净动作与 `x_start` 做比较来理解这个 loss。

## 8. 本文核验范围

除静态核对调用链外，使用该调试入口的全部覆盖参数合成 Hydra 配置，只将设备改为 CPU，以 4 条合成样本验证了实际 `Diffusion.loss()`：固定 `t=[0,5,12,23]` 和噪声后，其输出与手工计算的噪声 MSE 一致；张量形状与本文一致，反向传播能为全部可训练参数产生有限梯度。上面的加噪系数表也由仓库实际调度函数计算。

该核验没有读取完整专家数据集，也没有启动训练或仿真；它验证的是本文解释的实际模型前向、损失与梯度路径，不代表一次完整训练运行的结果。

补充核验了推理链：按相同入口配置、仅改为 CPU，使用合成状态、随机初始化网络参数和人为设定的动作裁剪边界，实际调用 `p_sample_loop(return_diffusion=True)`。确认内层网络恰好调用 24 次，索引为 `23→0`，单条观测的状态/动作形状分别为 `[1,20]`、`[1,3]`，全链形状为 `[1,25,3]`，且 24 步复用同一条件状态。固定 `t=0` 的输入重复采样，结果不受该步新采样噪声影响。此核验用于确认执行流程，没有使用训练好的检查点评价动作质量。

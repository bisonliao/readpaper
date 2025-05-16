**Proximal Policy Optimization Algorithms**

### Introduction

PPO方法希望解决：

1. RL方法是可扩展的（支持大模型、可以并行的训练）
2. 样本数据利用率高
3. 健壮稳定（能成功处理各种任务而不需要调超参数）

论文把PPO与TRPO做对比：PPO和TRPO一样具备样本的高利用率和可靠的性能，同时又不会像TRPO那样复杂的需要二阶导。



### Algorithm

下图是如何从普通的策略梯度，推出PPO的梯度：

![image-20250428170541008](img/image-20250428170541008.png)

算法如下：

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



### Experiments

下图是在一系列任务上的与其他算法的性能对比

![image-20250428172038764](img/image-20250428172038764.png)
### 目标：

1. 复现lift/can/square等等经典任务的训练
2. 使用robomimic这个现成的IL/RL库，完成自定义的环境/任务的训练

### 一、关键tips

#### 1、被验证过的超参数

下载数据后，运行 `generate_paper_configs.py` 会生成复现实验所需的配置文件。

```
# Assume datasets already exist in robomimic/../datasets folder. Configs will be generated under robomimic/exps/paper, and training results will be at /tmp/experiment_results when launching training runs.
$ python generate_paper_configs.py --output_dir /tmp/experiment_results

# Alternatively, specify where datasets exist, and specify where configs should be generated.
$ python generate_paper_configs.py --config_dir /tmp/configs --dataset_dir /tmp/datasets --output_dir /tmp/experiment_results
```

详细见文档：

```
https://robomimic.github.io/docs/datasets/robomimic_v0.1.html#reproducing-results
```

这个太重要了。执行后生成了一堆配置文件，分门别类非常清晰。

#### 2、robomimic与其他几个“库”的联系和区别

| 名称               | 它是什么                                                     | 主要作用                                                     | 与其他项目的关系                                             |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **robosuite**      | 基于 MuJoCo 的机器人操作仿真环境。【仿真环境与任务】         | 提供机械臂、物体、相机、动力学和 Lift、Can、Square、Transport 等任务 | RoboTurk 可在其中采集示范；MimicGen 可在其中生成数据；robomimic 可用这些数据训练并回到环境 rollout |
| **RoboTurk Pilot** | 使用 RoboTurk 遥操作平台众包采集的早期人类机器人示范数据集/实验。【人类遥操作采集工具】 | 研究如何从大量、不同水平的人类操作者示范中学习               | 数据通常在 robosuite 环境中采集；是 robomimic 人类示范研究的重要前身 |
| **MoMaRT**         | 面向多任务、长时序机器人操作的离线数据集与 benchmark。【数据集/benchmark】 | 包含成功、失败及不同质量的机器人轨迹，用于研究长时序任务和 mixed-quality data | 可由 robomimic 加载训练；任务通常建立在 robosuite 生态上     |
| **D4RL**           | 通用 offline RL 数据集与 benchmark 集合。【数据集/benchmark】 | 提供 MuJoCo locomotion、AntMaze、Adroit、Franka Kitchen 等固定离线数据 | 不是仿真器或训练库；robomimic 可支持其中部分机器人数据，但 D4RL 本身覆盖的领域更广 |
| **MimicGen**       | 从少量人类示范自动生成大量新示范的数据生成系统。             | 将示范分段，并根据新物体位置变换、拼接和执行轨迹             | 常在 robosuite 中生成数据；生成的数据可交给 robomimic 的 BC、IQL 等算法训练 |
| **robomimic**      | 机器人 imitation learning 和 offline RL 的数据、算法与评测框架 | 统一训练 BC、BC-RNN、Diffusion Policy、IQL、CQL、BCQ、TD3+BC 等算法 | 它消费 RoboTurk、MoMaRT、D4RL、MimicGen 等来源的数据，并可调用 robosuite 做 rollout 评估 |

一句话记忆：

> **robosuite 搭世界，RoboTurk 让人演示数据，MimicGen 扩数据，MoMaRT/D4RL 提供现成数据，robomimic 负责用数据训练策略。**

### 二、发散讨论

#### 1、是否可以继续用online RL微调

```
robomimic聚焦于IL/ offline RL算法，他们训练都不需要与环境交互。那么这些算法训练得到的agent，是否可以继续用online RL算法通过与环境交互进一步微调？ 如果可以，有下面的具体问题：robomimic提供的10个算法中，哪些可以无缝的直接转为online RL继续微调、转为哪个算法？ 哪些因为IL/offline RL训练得到的agent因为缺乏某种组件（例如Q网络、V网络没有）不能直接转online RL？这种情况下有没有可能补齐缺乏的组件而继续微调？会不会效果不好？
```



这是一个真正的 offline-to-online 流程：

```
robomimic 离线训练
    ↓
保存 checkpoint
    ↓
恢复 actor / critic / value / target network
    ↓
把环境新采集的数据放入 replay buffer
    ↓
继续进行 online RL 微调
```

在这个定义下，问题的核心确实是：checkpoint 中是否已经包含 online RL 所需的网络组件，以及这些组件能否和目标 online 算法的网络定义兼容。

先给结论：

- 所有算法都可以至少转移 policy，作为 online actor 的初始化；
- BCQ、CQL、IQL、TD3-BC、IRIS 已经包含部分或完整的价值网络，适合 offline-to-online；
- TD3-BC 最容易转成 TD3；
- IQL 最适合继续做 IQL-style online fine-tuning；
- CQL 最适合转成 SAC / online CQL；
- BC、BC-RNN、BC-Transformer、Diffusion Policy、HBC 缺少标准 critic，需要补充；
- 缺少的 critic 可以先用离线数据预训练，再开始 online RL，通常比随机初始化更稳；
- 即使所有组件都存在，也不能保证“直接加载后就稳定”，因为不同算法的 critic 目标和正则项可能不同。

逐个分析如下：

| robomimic 算法   | checkpoint 中主要组件                                    | 缺少的 online RL 组件                                 | 推荐衔接算法                            | 转移难度 | 结论                                  |
| ---------------- | -------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------- | -------- | ------------------------------------- |
| BC               | policy                                                   | Q、V、target network                                  | TD3 / SAC / PPO                         | 高       | 只能转移 actor                        |
| BC-RNN           | RNN policy                                               | recurrent Q/V、target network                         | Recurrent TD3 / Recurrent SAC           | 高       | actor 可转移，critic 需补             |
| BC-Transformer   | Transformer policy                                       | context-aware critic、target network                  | Transformer actor-critic                | 很高     | 可转移 policy，但需要专门 critic      |
| Diffusion Policy | diffusion actor                                          | Q/V、diffusion-compatible actor objective             | Diffusion actor-critic                  | 很高     | 只能作为 policy prior 或初始化        |
| HBC              | low-level actor、high-level planner                      | 高低层 critic、target value                           | Hierarchical SAC / hierarchical TD3     | 高       | planner 和 actor 可转移，critic 需补  |
| IRIS             | actor、planner、Q/value、target components               | online replay、探索机制、可能的 online target 逻辑    | Online hierarchical IQL / IRIS-style RL | 中高     | 组件较完整，但层级语义复杂            |
| BCQ              | Q critic、target critic、action sampler、可选 actor      | 若 actor disabled，则缺少 online actor                | Online BCQ / TD3-like                   | 中       | 价值组件较完整，需检查 actor 是否启用 |
| CQL              | actor、Q critic、target critic、可能的 entropy parameter | online SAC 需要的温度和目标逻辑可能不同               | SAC / online CQL                        | 中       | 很适合作为 actor-critic 初始化        |
| IQL              | actor、Q critic、target critic、V                        | 若转 SAC，需要 entropy temperature；没有 target actor | Online IQL / SAC-like                   | 中低     | 最适合做 offline-to-online 之一       |
| TD3-BC           | actor、target actor、Q critic、target critic             | 基本没有核心网络缺口                                  | TD3                                     | 最低     | 最容易转成 online RL                  |

##### offline-to-online方面比较有影响力的论文/工作：

| 工作                                                         | 论文链接                                                     | 核心观点                                                     | 主要 benchmark / 实验平台                                    | 与你想法的关系                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| AWAC: Accelerating Online Reinforcement Learning with Offline Datasets（2020） | [论文](https://arxiv.org/abs/2006.09359) · [项目页](https://awacrl.github.io/) | 用 advantage-weighted maximum-likelihood policy update 连接 offline 和 online 阶段。离线数据用于预训练，在线数据用于改进；相比直接 SAC，AWAC 更能避免初期性能下降并利用示范数据。 | D4RL Gym locomotion；MuJoCo 灵巧手任务；Sawyer 抽屉、D’Claw 阀门、灵巧手等真实机器人任务。论文在 Walker2d medium 上报告 offline-to-online 改进约 30.1 → 79.1 normalized score。[实验详情](https://arxiv.org/abs/2006.09359) | 最早期、最经典的 offline-to-online 机器人 RL 之一。它已经说明“保留适度行为约束、但允许在线策略改进”比直接 SAC 更有效。 |
| Offline-to-Online RL via Balanced Replay and Pessimistic Q-Ensemble / Off2On（CoRL 2021/2022） | [论文页](https://proceedings.mlr.press/v164/lee22d.html) · [代码](https://github.com/shlee94/Off2OnRL) | 发现直接 online fine-tuning 会因为 state-action distribution shift 产生 bootstrap error，破坏离线策略。方法使用 balanced replay，并从离线阶段转移 pessimistic Q ensemble。 | D4RL locomotion、AntMaze，以及多个机器人操作任务。           | 对你的问题非常直接：它研究了为什么“加载 offline critic 后直接在线训练”会崩，以及为什么 replay 和 critic transfer 都重要。 |
| Efficient Online RL with Offline Data / RLPD（ICML 2023）    | [论文](https://arxiv.org/abs/2302.02948) · [代码](https://github.com/ikostrikov/rlpd) | 不再设计复杂的离线约束，而是对 SAC 做少量关键修改：离线/在线数据对称采样、较大的 critic ensemble、LayerNorm、较高 update-to-data ratio。核心观点是正确的 replay 和 critic regularization 比复杂 constraint 更重要。 | D4RL locomotion、AntMaze、Adroit，以及 pixel control。论文报告相对已有方法最高约 2.5× 改善。[实验说明](https://arxiv.org/abs/2302.02948) | 对你“最终摒弃 offline constraint”的想法很重要。RLPD 已经提供了一个强基线：不保留复杂 offline constraint，而使用 balanced replay 和稳定 critic。 |
| Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning（NeurIPS 2023） | [论文](https://arxiv.org/abs/2303.05479) · [代码](https://github.com/nakamotoo/Cal-QL) | 让 offline Q function 保持“校准”：对当前策略提供保守下界，同时维持合理数值尺度，使它适合 online fine-tuning。方法基于 CQL，论文强调只需很小的实现改动即可转为 Cal-QL。 | D4RL AntMaze 和 Adroit；共 11 个 fine-tuning 任务，在 9/11 上优于对比方法。[官方代码说明](https://github.com/nakamotoo/Cal-QL) | 它证明“offline 阶段的 critic 是否适合 online 阶段”是关键问题。你的组件迁移研究可以把 Cal-QL 作为 critic calibration baseline。 |
| Policy Expansion for Bridging Offline-to-Online RL / PEX（ICLR 2023） | [论文](https://arxiv.org/abs/2302.00935) · [代码](https://github.com/Haichao-Zhang/PEX) | 不直接修改或破坏 offline policy，而是保留 offline policy，同时新增 online policy；两者根据 Q 值自适应组合。论文还系统消融了 policy transfer、critic transfer、offline buffer 和 offline policy freeze。 | D4RL AntMaze、HalfCheetah、Hopper、Walker2d；每个任务约 1M offline steps + 1M online steps。[任务表](https://arxiv.org/abs/2302.00935) | 这是对“完全删除 offline 约束”最直接的反例：不是把旧策略约束逐渐删掉，而是冻结旧策略并添加新的 online policy。你的研究必须明确比较 constraint release 与 policy preservation。 |
| Jump-Start RL / JSRL（ICML 2023）                            | [论文](https://arxiv.org/abs/2204.02372) · [项目页](https://jumpstart-rl.github.io/) | 将已有 policy、示范或 offline 数据转化为一个“指导策略”，给 online RL 提供 curriculum / jump-start 行为；核心目标是避免从随机策略开始探索。 | D4RL AntMaze 和 Adroit；offline 1M transitions，online fine-tuning 1M steps。[实验设置](https://proceedings.mlr.press/v202/uchendu23a/uchendu23a.pdf) | 它把 offline policy 看作 online learning guide，而不只是 actor 初始化。对于 BC-RNN 这类没有 critic 的策略，JSRL 提供一种“保留 policy、另训 online policy”的思路。 |
| MOTO: Offline Pre-training to Online Fine-tuning for Model-based Robot Learning（CoRL 2023） | [论文](https://arxiv.org/abs/2401.03306) · [项目页](https://sites.google.com/view/mo2o/) | 针对高维视觉机器人任务，提出 model-based offline-to-online 方法。通过 model-based value expansion、policy regularization 和 epistemic uncertainty 控制，减少模型 exploitation 和分布偏移。 | 10 个 MetaWorld 任务；Franka Kitchen 图像任务。论文声称在 9/10 MetaWorld 任务上具有强性能，并解决了图像 Franka Kitchen。[论文详情](https://proceedings.mlr.press/v229/rafailov23a/rafailov23a.pdf) | 与你的视觉 BC-RNN/Square 方向最接近的高维机器人工作之一，但它使用 model-based RL，和你想研究的 robomimic model-free checkpoint 迁移形成互补。 |
| Pre-Training for Robots / PTR（2022）                        | [论文](https://arxiv.org/abs/2210.05178)                     | 使用大规模多任务机器人数据进行 offline CQL pretraining，再用少量目标任务数据进行 fine-tuning；重点是让 offline RL 充当机器人策略预训练器。部分实验包括少量在线/自主 fine-tuning，但核心贡献更多是 offline pretraining + target-data adaptation。 | BridgeData、多任务真实机器人数据、WidowX；新任务可只用约 10–15 条目标任务示范。[论文说明](https://arxiv.org/abs/2210.05178) | 它不是最严格意义上的纯 online O2O 基准，但对你的方向很重要：说明“不同 offline checkpoint 的可迁移性”本身可以成为研究问题。 |
| Imitation Bootstrapped RL / IBRL（RSS 2024）                 | [RSS论文](https://www.roboticsproceedings.org/rss20/p056.html) · [arXiv](https://arxiv.org/abs/2311.02198) · [项目页](https://ibrl.hengyuanhu.com/) | 先用示范训练 IL policy，再让 IL policy 提供 online exploration action proposal 和 Q target bootstrap。它不是简单把 BC loss 当 regularizer，而是让高质量 IL 动作直接参与在线探索和价值目标。 | 6 个仿真任务、3 个真实机器人任务；包括 MetaWorld 和 robomimic 场景，项目页明确报告了 Square 等 robomimic 任务。[项目结果](https://ibrl.hengyuanhu.com/) | 这是与你的 BC/BC-RNN → online RL 想法最接近的工作之一。它说明只有 actor 的 IL policy 也可以通过 action proposal 帮助 online critic，而不必直接把 actor 改成 SAC actor。 |
| RoboFuME: Robot Fine-Tuning Made Easy（2023/2024）           | [论文](https://arxiv.org/abs/2310.15145) · [IEEE页面](https://ieeexplore.ieee.org/document/10610421/) | 用多任务离线数据预训练策略，用 VLM 作为 reward classifier，进行 reset-free online fine-tuning，减少人工奖励标注、重置和干预。 | 5 个真实机器人操作任务：糖果清扫、折叠布、覆盖方块、锅盖和 pot pick-and-place；约 3 小时自主真实机器人经验。[论文摘要](https://arxiv.org/abs/2310.15145) | 它把 offline-to-online 推向真实机器人系统，说明除了网络组件和算法约束，reward、reset 和安全采样也是 O2O 的核心问题。 |

#### 2、怎么理解不同离线数据来源对IL/offline RL的影响，是不是质量越高越好

以 100 条不同水平的人类示范为例：

| Dataset | BC Can | BC-RNN Can | BC Square | BC-RNN Square |
| ------- | ------ | ---------- | --------- | ------------- |
| Worse   | 56.7   | 92.0       | 22.0      | 39.3          |
| Okay    | 72.0   | 95.3       | 27.3      | 45.3          |
| Better  | 83.3   | 99.3       | 58.7      | 66.0          |

在数据量相同的情况下，`Better > Okay > Worse` 基本成立。

所以 robomimic 支持：

> 更高质量示范通常更好。

它发现的反常点是：加入额外的 Worse demonstrations 后，大部分方法会退化，但 BC-RNN 经常继续改善。

例如 Square：

| Dataset          | 数量 | BC-RNN success |
| ---------------- | ---- | -------------- |
| Better           | 100  | 66.0           |
| Worse + Better   | 200  | 73.3           |
| Okay + Better    | 200  | 74.0           |
| Proficient Human | 200  | 84.0           |

这说明：

- `100 Better + 100 Worse` 优于 `100 Better`；
- 但仍不如 `200 Proficient Human`。

因此不能说“低质量优于高质量”。更准确的是：

> 当高质量数据数量有限时，额外的低质量数据仍可能提供正价值；但等量的高质量数据通常更好。

##### “低质量”本身也不是一个标量

robomimic 的 `Worse` 主要表示操作者：

- 动作更慢；
- 轨迹更长；
- 更容易出现无效移动和局部错误；
- 操作策略更不一致。

它们不一定都是失败轨迹。很多 `Worse` demonstrations 最终仍然成功，只是完成得不够漂亮。

`Can-Paired` 才是真正包含同一初始状态下成功和失败轨迹的数据。

因此至少要拆开：

| 维度                      | “低质量”的可能含义                        |
| ------------------------- | ----------------------------------------- |
| Outcome quality           | 最终失败或 return 低                      |
| Action efficiency         | 成功但路径长、动作抖动、绕路              |
| Local action quality      | 某些时间步采取明显错误动作                |
| Demonstrator consistency  | 相似状态下动作不一致                      |
| Temporal coherence        | 延迟、停顿、反复修正                      |
| Recovery content          | 是否包含出错后的纠正                      |
| State coverage            | 是否访问 expert 没访问过的状态            |
| Information compatibility | demonstrator 是否使用 policy 看不到的信息 |

一条轨迹可以同时是：

```
低效率
+ 最终成功
+ 覆盖了有用的失败状态
+ 包含高价值 recovery
```

把它简单标为“low quality”会丢失最重要的信息。

##### 低质量数据为什么可能有帮助

###### 1. 扩大状态覆盖

Expert 通常沿着狭窄的成功路径行动：

```
接近 → 抓取 → 搬运 → 放置
```

实际 policy 部署后会偏离这条路径。低质量 demonstrations 可能包含：

```
抓偏 → 调整 → 重新抓取 → 成功
```

这些状态对训练闭环 policy 很重要，因为它们告诉机器人偏离正常路径后怎么办。

###### 2. 提供 recovery 行为

单纯 expert data 经常只告诉 policy：

> 正确的时候下一步怎么做。

Recovery data 还能告诉它：

> 已经做错的时候，怎样恢复。

这对 IL 特别重要，因为 BC 的典型问题就是 covariate shift：训练时只见过正确状态，部署时一旦犯错就进入未见状态。

###### 3. 增加任务和动作多样性

不同操作者可能使用不同但都合理的策略。

这可能帮助模型泛化，但也要求 policy 能建模 multimodality。普通 MSE BC 容易把两个合理动作平均成一个不合理动作；BC-RNN、GMM 或 Diffusion Policy 更有机会利用它。

###### 4. 给 offline RL 提供价值对比

理论上，包含好坏行为的数据可以告诉 critic：

```
这个状态下，动作 A 最终成功；
动作 B 最终失败。
```

但需要同时满足：

- reward 和 done 标注正确；
- 相似状态下同时出现好动作和坏动作；
- critic 能正确识别状态；
- 数据覆盖足够；
- 没有严重 partial observability。

robomimic 的 Can-Paired 说明，当时的 BCQ/CQL 即使拿到这种数据也没能很好地利用。这说明算法失败，不代表失败数据没有信息。

###### 5. 帮助视觉表示学习

更多轨迹意味着更多物体位置、遮挡、视角和机械臂姿态。即使部分动作不理想，图像仍可以帮助 encoder 学到更广的视觉分布。

##### 为什么 machine mixed-quality 容易被 offline RL 利用，而 human mixed-quality 不容易

这是 robomimic 最值得深入的发现之一。

| Machine-generated data                             | Human-generated data                             |
| -------------------------------------------------- | ------------------------------------------------ |
| 通常来自同一种 RL policy family 的不同 checkpoints | 来自不同人的策略、习惯和意图                     |
| 状态和动作更接近 Markov 设定                       | 动作可能依赖过去操作和人的私有意图               |
| 动作频率和控制延迟一致                             | 存在 teleoperation delay、停顿和修正             |
| suboptimality 通常表现为 policy competence 不同    | 同时包含 competence、style、noise、multimodality |
| reward 与生成 policy 的目标一致                    | 人可能使用 reward 没有表达的策略偏好             |
| exploration 产生相对连续的 coverage                | 不同人的 coverage 可能割裂                       |

所以不能简单地说：

> Offline RL 能处理 mixed-quality machine data，因此也应该能处理 mixed-quality human data。

这两种“mixed quality”并不是同一种数据分布。

##### 行业内的相关研究

| 核心问题                                              | 直接先例                                                     |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| Human/machine、质量、规模对 IL/offline RL 的影响      | [What Matters in Learning from Offline Human Demonstrations for Robot Manipulation](https://arxiv.org/abs/2108.03298)（CoRL 2021） |
| 数据质量与 state-action coverage 如何影响 offline RL  | [A Dataset Perspective on Offline Reinforcement Learning](https://arxiv.org/abs/2111.04714)（CoLLAs 2022） |
| 什么条件下 offline RL 优于 BC                         | [When Should We Prefer Offline Reinforcement Learning Over Behavioral Cloning?](https://arxiv.org/abs/2204.05618)（ICLR 2022） |
| IL 数据质量是多维的                                   | [Data Quality in Imitation Learning](https://arxiv.org/abs/2306.02437)（2023） |
| 广覆盖数据可能比算法复杂度更重要                      | [Don't Change the Algorithm, Change the Data: Exploratory Data for Offline Reinforcement Learning](https://arxiv.org/abs/2201.13425)（ExORL，2022） |
| 真实机器人数据来源对 BC/offline RL 的影响             | [Real World Offline Reinforcement Learning with Realistic Data Source](https://arxiv.org/abs/2210.06479)（ICRA 2023） |
| Human play、scripted data、视觉和 stitching benchmark | [D5RL: Diverse Datasets for Data-Driven Deep Reinforcement Learning](https://arxiv.org/abs/2408.08441)（RLC 2024） |
| Generated data 与 human data 的关系                   | [MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations](https://arxiv.org/abs/2310.17596)（CoRL 2023）；[Tether: Autonomous Functional Play with Correspondence-Driven Trajectory Warping](https://arxiv.org/abs/2603.03278)（ICLR 2026） |
| 删除次优和冗余 demonstrations                         | [SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning](https://arxiv.org/abs/2505.22626)（2025） |
| Recovery-rich data collection                         | [Adversarial Data Collection: Human-Collaborative Perturbations for Efficient and Robust Robotic Imitation Learning](https://arxiv.org/abs/2503.11646)（2025） |
| Privileged expert 不一定可模仿                        | [Robust Asymmetric Learning in POMDPs](https://proceedings.mlr.press/v139/warrington21a.html)（ICML 2021）；[Causal Imitation Learning with Unobserved Confounders](https://arxiv.org/abs/2208.06267)（2022） |
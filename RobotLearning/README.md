### 算法

| 算法名           | 一句话描述                                                   | 分类               | RL backbone                                                 |
| ---------------- | ------------------------------------------------------------ | ------------------ | ----------------------------------------------------------- |
| ACT              | 策略一次预测接下来 k 个时间步的目标关节位置，而非每次只预测一步；同时采用 CVAE 架构，其 encoder 和 decoder 均由 Transformer 构成。 | IL                 | DDPG                                                        |
| A-LIX            | 基于像素输入的离策略强化学习在训练早期因TD目标含噪，使CNN编码器产生空间不连续的特征梯度，导致训练不稳定；A-LIX通过双线性插值在编码器输出特征图上进行自适应局部混合，平滑梯度并稳定训练。 | 离策略的online RL  | DDPG（连续动作空间），DQN（离散动作空间）                   |
| C2FQN            | C2FQN将连续动作空间逐层离散化，通过价值型RL反复选择最高Q值区间进行细化，最终得到高精度动作。 | 离策略的online RL  | DQN类算法                                                   |
| Dreamer          | Agent与环境交互的数据用于学习世界模型；再在其潜空间模拟器中通过想象轨迹进行RL训练改进策略，并用策略产生的新交互数据持续迭代更新世界模型与策略。 | Model-Based RL     | 具有明显的 replay-based off-policy 特征                     |
| Dreamer V3       | Dreamer精细微调版本                                          |                    |                                                             |
| DrM              | 神经网络中休眠神经元的比例，是衡量智能体是否学会有效技能的重要指标。DrM利用该比率：比率高时强化探索，并周期性对网络权重施加随机扰动；比率低时则转向利用。 | 离策略的online RL  | DDPG                                                        |
| DrQ              | DrQ在RL训练更新网络时，对视觉观测进行随机shift增强，并约束同一观测的不同变换具有相近的Q值，从而提升视觉RL训练稳定性并减少陷入次优解。 | 离策略的online RL  | 插件化，可用于多种 model-free off-policy 算法，例如SAC、DQN |
| DrQ v2           | 在 DrQ 基础上同时修改了 RL backbone、target return、图像增强、探索策略、关键超参数和底层实现。 | 离策略的online RL  | DDPG                                                        |
| Implicit QL      | 离线 RL 若用标准 Q-learning 提升行为策略，就要对数据外动作外推 Q 值，容易因最大化偏差高估而选错动作。IQL 只在数据动作上计算 Q，用非对称平方损失学习其高 expectile 作为 \(V(s)\)，再用 \(r+ gamma * V(s')\) 进行多步更新，最后按优势加权模仿高价值动作。 | offline RL         | 是一个独立的离线 RL 算法                                    |
| Conservative QL  | 类似IQL面临的问题，CQL对Q网络的损失函数引入一个正则项：CQL保守项，它会惩罚所有可能动作的Q值，防止高估，同时它会保护离线数据集内的动作，允许他们的Q值相对较高。它可以基于所有基础的价值离策略RL方法，例如DQN  SAC等等 | offline RL         | SAC/DQN/<br />TD3/DDPG                                      |
| TD3+BC           | 类似IQL面临的问题，该方法在TD3算法的基础上对损失函数增加行为克隆正则项，把TD3算法从online RL改造成了offline RL算法。 | offline RL         | TD3                                                         |
| BCQ              | 类似IQL面临的问题，BCQ的核心是限制TD目标 y=r+gamma* max_a[Q(s', a)]中的动作 aa 必须接近数据集动作分布：用VAE生成数据分布中的候选动作，再经扰动网络小幅调整，最后由target Q网络从候选动作中选Q值最高者，从而避免对大量OOD动作进行Q值评估。BCQ 并不是要求动作必须是数据集中原封不动出现过的动作，而是通过 VAE + perturbation 生成与数据分布接近的动作。 | offline RL         | DQN/TD3                                                     |
| BC               | 完全基于离线数据，对agent输出的动作进行监督学习              | IL                 | 可预训练policy-based方法和DQN                               |
| DAgger           | N轮训练，每轮分为 与环境交互、专家标注出现状态的action、有监督学习三步；随着轮数的增加，专家与环境交互的比例降低、被训练的agent本身与环境交互的比例提高。标注训练数据，训练数据里每个状态下应该出现什么动作，都是专家标注的；而训练数据里有哪些状态，主要是学生模型探索出来的。 | IL                 | 可预训练policy-based方法                                    |
| MWM              | 在Dreamer的基础上，把视觉表示学习和潜在动力学模型的学习两部分解耦开来，获得比Dreamer更好的性能。 | Model-Based RL     | Dreamer                                                     |
| RLPD             | RLPD 以 SAC 等在线 off-policy RL 为基础，在每个 batch 中从在线 replay buffer 和固定离线数据各采样 50% 的 transitions，通过 critic 中的 LayerNorm 抑制对数据分布外动作的灾难性 Q 值外推，通过提高 UTD ratio 让离线数据被更充分地 Bellman backup，并利用大规模 critic ensemble 与随机抽取少量 target critic 构造目标来正则化高频更新、降低过拟合，从而在保留在线探索能力的同时提高样本效率。。 | 离策略的online RL  | SAC                                                         |
| Diffusion Policy | Diffusion Policy 是一种条件扩散式模仿学习方法：它以专家示范的真实动作序列为训练数据，将观测序列作为条件，把随机选取噪声等级 k后得到的带噪动作序列 与 k输入网络，预测所加入的噪声，并以预测噪声与真实噪声之间的 MSE 进行监督训练；推理时则从随机动作噪声出发，逐步去噪生成动作序列。 | imitation learning |                                                             |
| Octo             | 一个开源的、通用的机器人操作、基于 Transformer 的策略，在来自 Open X-Embodiment 数据集的 80 万个episode上进行IL预训练。它支持灵活的任务和观察定义，并且可以快速微调以适应新的观察和动作空间。使用Octo初始化模型并在下游场景进行微调的时候，可以是IL，也可以是offline / online RL。 | imitation learning |                                                             |
| OpenVLA          | OpenVLA 是一个拥有 7B 参数的开源视觉-语言-动作模型（VLA），在 Open X-Embodiment 数据集约 970k 个真实机器人操作轨迹上进行大规模行为克隆（IL/BC）训练；它支持多种机器人开箱即用，并可通过参数高效微调快速适应新的机器人场景。OpenVLA 的 checkpoint 和 PyTorch 训练流程均开源，模型可从 Hugging Face 下载并微调；微调既可以采用 IL/BC，也可以作为预训练策略进一步用于 Offline/Online RL。 | imitation learning |                                                             |
| AWAC             | AWAC 用离策略 TD 学习估计当前策略的 Q 值，再对经验池中的动作进行优势加权模仿，从而利用离线数据预训练，并通过在线交互持续改进策略。 | Off2OnRL           | SAC风格                                                     |
| IBRL             | IBRL 先用离线示范数据训练并固定一个独立的模仿策略，再在在线 RL 训练中，让模仿策略与 RL 策略分别提出候选动作，由 critic 选择 Q 值较高者用于环境交互和 TD 目标计算，从而借助模仿策略加速探索与价值学习，并通过在线交互实现超越示范的策略改进。 | Off2OnRL           | SAC/TD3/DDPG                                                |
| PTR              | Pre-Training for Robots解决在兼容的观测与动作接口、统一的奖励尺度下，机器人在多任务间的迁移(例如都是厨房系列的不同任务)。先通过大量多任务离线强化学习，从已有示范中学习可迁移的策略和价值判断（离线预训练），再用少量目标任务示范适应新任务（微调阶段可以是offline RL或者online RL）；但让这条路线有效，网络结构、奖励标注和微调方式都很关键。 | Off2OnRL           | CQL                                                         |
| RoboFuME         | 在CalQL算法基础上，利用已有机器人示范数据预训练控制策略，同时将预训练视觉语言模型微调成奖励模型，再让机器人交替执行目标任务和重置任务，在少量人工干预下自主练习、持续改进。重点解决了微调阶段真实环境下的人工重置工作和奖励设计工作 | Off2OnRL           | CalQL                                                       |



### algorithm libraries

| Libraries                        | 主要研究问题                                                 | 官方/官方关联代码中的代表性算法                              |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ManiSkill 3**                  | 通用机器人 RL、视觉 RL、dexterous/tabletop/mobile manipulation | **PPO、SAC、TD-MPC2**；BC、Diffusion Policy、ACT、RFCL、RLPD、SAC+Demos |
| **Bi-DexHands / DexterousHands** | 高维双手灵巧操作、MARL、offline RL、multi-task/meta-RL       | **PPO、TRPO、DDPG、TD3、SAC、MAPPO、HAPPO、HATRPO、IPPO、MADDPG、BCQ、TD3+BC、IQL、MTPPO、MTSAC、MTTRPO、MAML、ProMP** |
| **robobase**                     | 覆盖了机器人训练用到的Offline-RL/IL                          | **DrQv2 / ALIX /SAC_LIX / DrM / dreamer v3 / MWM / CQN / IQL_DrQv2 / diffusion / ACT** |
| **robomimic**                    | Robot Learning from Demonstration / Offline RL               | **BC、BC-RNN、BC-Transformer、Diffusion Policy、HBC、IRIS、BCQ、CQL、IQL、TD3+BC** |

### 环境

| Benchmark                   | 主要研究问题                                            | 任务 / 数据集规模                                            | 机器人 / 环境                 | 主要评测方向                                                 |
| --------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------- | ------------------------------------------------------------ |
| **Meta-World**              | Multi-task RL、Meta-RL、task generalization             | **50 个 manipulation tasks**；ML1 / ML10 / ML45 / MT10 / MT50 | MuJoCo；主要是 Sawyer         | 单任务、多任务、跨任务泛化、meta-learning                    |
| **RLBench**                 | 通用 robot manipulation、task generalization            | **100 个任务**，每个任务支持大量/无限 demonstrations         | Franka Panda + CoppeliaSim    | imitation learning、RL、multi-task、视觉 manipulation        |
| **LIBERO**                  | Lifelong / continual robot learning、knowledge transfer | **130 个任务**：LIBERO-Spatial 10、Object 10、Goal 10、100   | Franka Panda + tabletop       | spatial/object/goal knowledge transfer、lifelong learning    |
| **CALVIN**                  | Language-conditioned、long-horizon manipulation         | **34 个任务**，4 个环境 A/B/C/D；长序列语言任务              | Franka Panda                  | language-conditioned policy、long-horizon、multi-task/generalization |
| **D4RL – Adroit**           | Offline RL / imitation learning                         | **4 个任务**：Pen、Door、Hammer、Relocate；每个有 demos / cloned / expert 数据集 | 24-DoF Adroit hand            | Offline RL、IL、demonstration learning                       |
| **D4RL – FrankaKitchen**    | Offline RL / multi-task manipulation                    | Kitchen manipulation tasks；complete / partial / mixed 等数据 | Franka                        | Offline RL、multi-task、compositional manipulation           |
| **FurnitureBench**          | 长时序真实机器人操作、assembly                          | 家具装配任务；**200+ 小时 teleoperation data**，同时有 FurnitureSim | Franka / real-world setup     | Long-horizon manipulation、assembly、sim-to-real             |
| **MimicGen**                | 大规模 demonstration generation、IL                     | **48,000+ demonstrations / 12 tasks**；不同物体、机器人、reset distribution | robosuite 中多种机器人        | imitation learning、data scaling、generalization             |
| **BridgeData V2**           | Large-scale real-world robot learning                   | **60,096 trajectories / 24 environments / 13 skills**        | WidowX 250                    | multi-task、open-vocabulary、cross-environment generalization |
| **Open X-Embodiment (OXE)** | Cross-robot / cross-embodiment learning                 | **1M+ trajectories / 60 datasets / 22 robot embodiments / 527 skills** | 单臂、双臂、quadruped 等      | cross-embodiment learning、generalist policies、VLA pretraining |
| **RoboCasa / RoboCasa365**  | Generalist robot、everyday household manipulation       | RoboCasa365：**365 tasks、2,500+ kitchen scenes、3,200+ objects、2,200+ hours robot demonstrations** | Franka / kitchen environments | generalist policy、multi-task、large-scale simulation        |
| **RH20T**                   | Contact-rich manipulation、multimodal robot learning    | **110K+ sequences / 147 tasks / 7 robot-gripper configurations** | 多种真实机械臂                | force/tactile/audio/RGB-D、多模态 manipulation               |
| **RoboTwin**                | Bimanual manipulation、sim-to-real、data generation     | **50 个双臂任务 / 731 objects**（RoboTwin 1.x）              | ALOHA 等双臂平台              | bimanual manipulation、data generation、sim-to-real          |

### 
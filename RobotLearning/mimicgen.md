

**MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations**

CoRL'2023, from NVIDIA and 德克萨斯大学

### **MimicGen 主要是数据生成系统，但不是“完全离线”的过程。**

它从少量人类示范出发，在仿真环境中不断生成新的轨迹，最后保存为 HDF5 离线数据集。生成阶段仍然会：

- 创建仿真环境；
- 重置场景；
- 执行机器人动作；
- 读取状态、图像、奖励和成功信号；
- 筛选成功轨迹。

因此更准确地说，它是“基于仿真交互的离线示范数据生成”，而不是离线 RL 算法。论文报告了从约 200 条人类示范生成超过 50K 条示范数据。[论文摘要](https://arxiv.org/abs/2310.17596)

### **它不是和 robomimic 同层次的算法库。**

可以这样区分：

| 项目      | 主要职责                                       |
| --------- | ---------------------------------------------- |
| MimicGen  | 自动生成、扩增机器人示范数据                   |
| robomimic | 使用 HDF5 数据训练 BC、BC-RNN、IQL、CQL 等策略 |

MimicGen 官方文档明确说明，生成的数据与 robomimic 兼容，可以直接使用 robomimic 进行策略训练，常见的是 BC-RNN。[Getting Started](https://mimicgen.github.io/docs/tutorials/getting_started.html)

所以 MimicGen 自己的“核心算法”是数据生成流程、子任务分解、轨迹选择和轨迹变换，并不提供像 robomimic 那样完整的 IL/RL 算法集合。

### **它的核心原理是“面向物体的子任务轨迹拼接和变换”。**

典型流程如下：

1. 准备少量源示范，例如人类遥操作的 `square` 任务。
2. 把一个长任务分解成有顺序的物体中心子任务，例如“抓取物体”“将物体放置到目标位置”。
3. 根据子任务终止信号，把每条示范切成连续片段。
4. 在一个新的场景中，为当前子任务选择一个源轨迹片段：
   - 随机选择；
   - 或根据当前物体姿态选择最近邻示范。
5. 将源片段中的末端执行器轨迹转换到新物体姿态下，同时保持末端执行器与物体之间的相对位姿关系。
6. 从当前机器人状态插值到新片段的起点，然后在仿真环境中执行。
7. 记录新的状态、观测和动作；失败轨迹可以丢弃，继续生成直到达到目标数量。

官方代码将这几个部分对应为：

- `DatagenInfo`：保存末端位姿、物体位姿、目标位姿和子任务信号；
- `SelectionStrategy`：选择源示范片段；
- `WaypointTrajectory`：变换和拼接末端执行器目标轨迹；
- `TaskSpec`：定义子任务顺序及其参数。[Codebase Overview](https://mimicgen.github.io/docs/modules/overview.html) · [Data Generation Classes](https://mimicgen.github.io/docs/modules/datagen.html)

它不是简单地对图像做增强，也不是直接复制动作序列，而是利用物体坐标系对控制轨迹进行几何变换。

### **可以适配自己的环境和任务，但需要实现接口。**

官方明确支持新任务和新的仿真框架。[Custom Environments](https://mimicgen.github.io/docs/tutorials/datagen_custom.html)

通常需要完成：

- 为你的仿真器实现一个基础 `MG_EnvInterface`；
- 为具体任务实现任务接口类；
- 提供相关物体的 4×4 位姿；
- 提供子任务完成信号，或使用人工标注切分示范；
- 实现环境动作与末端执行器目标位姿之间的转换；
- 定义 `TaskSpec`，指定子任务顺序、参考物体和选择策略；
- 确保环境可以通过 robomimic 的环境封装创建；
- 将源示范整理成 robomimic HDF5，再运行 `prepare_src_dataset.py`。

其中最关键的前提是：任务能够合理地分解为一系列**物体中心子任务**，并且每个子任务的运动主要由相关物体的位姿决定。官方接口要求任务能够提供物体位姿和子任务终止信号。[Environment Interfaces](https://mimicgen.github.io/docs/modules/env_interfaces.html)

结合你现在的 `square` 任务，最现实的路线是：

```
少量 square 源示范
        ↓
实现或复用 MG_Square 接口
        ↓
prepare_src_dataset.py
        ↓
配置 square TaskSpec
        ↓
MimicGen 生成大量成功轨迹
        ↓
robomimic BC-RNN / IQL 训练
```

你现有的 `demo_v15.hdf5` 不一定能直接用于 MimicGen。它虽然包含状态和动作，但还需要额外的物体位姿、子任务终止信号以及 MimicGen 所需的控制信息。


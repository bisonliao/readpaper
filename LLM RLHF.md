**Training language models to follow instructions with human feedback**

### Introduction

通过结合监督学习（人类演示数据）和强化学习（人类偏好排序数据）的微调方法（RLHF）, 可以使得大模型：

1. 更好的遵循人类的意图
2. 更好的生成真实的答案
3. 减少毒性输出
4. 有很好的泛化能力，没有参与微调标注工作的其他人也认为微调后的大模型的输出更符合他们的需要

![image-20250510133101639](img/image-20250510133101639.png)

Q：RLHF是否包括step1的SFT？

A:  

- **若问流程**：RLHF完整流程包括SFT → RM训练 → RL微调。
- **若问技术核心**：RLHF特指**利用人类反馈的强化学习部分**（RM+PPO），SFT是其前置准备。

需要在具体场景中明确语境，但多数实践（如ChatGPT/InstructGPT）默认RLHF包含SFT阶段 

### Method and Experimental Details

#### method概述：

- Step 1: Collect demonstration data, and train a supervised policy. Our labelers provide demonstrations of the desired behavior on the input prompt distribution (see Section 3.2 for details on this  distribution). We then fine-tune a pretrained GPT-3 model on this data using supervised learning.

- Step 2: Collect comparison data, and train a reward model. We collect a dataset of comparisons  between model outputs, where labelers indicate which output they prefer for a given input. We then  train a reward model to predict the human-preferred output.

- Step 3: Optimize a policy against the reward model using PPO. We use the output of the
RM as a scalar reward. We fine-tune the supervised policy to optimize this reward using the PPO  algorithm (Schulman et al., 2017).

Steps 2 and 3 can be iterated continuously; more comparison data is collected on the current best  policy, which is used to train a new RM and then a new policy. In practice, most of our comparison data comes from our supervised policies, with some coming from our PPO policies.

#### 数据集

三部分数据集：

1. SFT数据集，13k条，prompt来自用户提交的API，回答来自标注人手写
2. RM（奖励模型）数据集，33k条，来自API和标注人的标注
3. PPO数据集，31k条，只需要prompt，不需要任何标注

96%是英语。

下表是来自API的prompt的归类分布：

![image-20250510154539754](img/image-20250510154539754.png)

该项目对标注者有严格的考试筛选、入职培训、全程的工作讨论。并且有专门雇佣一拨人，只负责测试，不负责标注数据，以检验微调后的模型的人类偏好是否具有泛化性。

#### 模型

##### RM训练阶段

RM模型初始化，直接使用第一步SFT后得到的模型（6B大小）去掉最后一层（词汇表分类层），改为一个标量输出层，用来输出reward值。然后进行RM训练。

RM训练是这样构造的：

![image-20250510161817037](img/image-20250510161817037.png)

##### 强化学习阶段

![image-20250510163015403](img/image-20250510163015403.png)

**bandit环境，简单的说就是回合长度为1的即时奖励的环境。**



**强化学习阶段很重要的是避免过拟合到RM：**

![image-20250510171404186](img/image-20250510171404186.png)

### Result

![image-20250510191128730](img/image-20250510191128730.png)

### Discussion

局限性：

1. 方法论，依赖一小部分标注者的倾向，不完全代表所有用户
2. 模型方面，并没有彻底做到对齐人类的偏好或者彻底的安全
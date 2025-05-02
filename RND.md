**EXPLORATION BY RANDOM NETWORK DISTILLATION**

### Introduction

环境的稀疏奖励，对于强化学习来说是个挑战，尤其是当人工设计额外的奖励也不切实际的时候。这种情况下，以直接方式探索环境的方法是必要的。

该论文创造了一种鼓励探索的方法：

1. 容易实现
2. 对高维观测友好
3. 对各种策略优化算法适用
4. 计算高效（论文说只需要前向传播，我认为不正确，它还是会后向传播更新预测网络的权重）

亲自玩一下这个游戏就有切身体会：特别有必要对不一样的局面进行额外奖励才行，否则Agent会被困在原地：

1. 奖励稀疏、长时间没有奖励的问题：Agent必须想办法进入到另外的区域（不是简单游走就能实现，得尝试各种技巧）拿到钥匙，然后干点啥（我还没有做到），而在原地转圈、跳动、比较简单的跳走到旁边的平台，都不能实现最终目的、不能获得奖励。。
2. 部分可观测的问题：很多房间在地图上看不到，需要探索（星际争霸里地图一开始是黑的，观测野很小）
3. 奖励延迟巨大的问题：就算离开一个房间进入新房间是由奖励的，但在第几千步时刻拿到的钥匙，可能在第百万步的时候用来开门。很多游戏都有这个问题，例如简单的breakout，击中砖块可能是好多步之前托盘成功托举小球的结果

![image-20250430091114783](img/image-20250430091114783.png)

到目前为止，我见过的鼓励探索的方法有：

1. epsilon greedy方法

2. 对于输出概率的模型，在softmax之前，对logits乘以一个温度

3. 在损失函数里，加上 输出的动作分布的熵的正则项

4. 对输出的动作的概率分布，做随机抽样，而不是argmax

5. 对输出加上一个噪声，或者对神经网络的参数加噪声

6. 表格法或者MCTS中，访问计数字段用作分母，也是鼓励探索

7. alphaGo里用到的对抗自博弈也是一种探索

8. 专家演示经验用于训练是一种让探索更有效的工作，例如alphaGo的前期的监督学习就使用了专家棋谱

9. 网络输出一个概率分布而不是确定性的价值/动作，也是一种鼓励探索，例如C51

10. 本文RND方法提到的随机网络蒸馏的方法

    



### Method

openai提供了配套的[代码](https://github.com/openai/random-network-distillation)

![image-20250429160415835](img/image-20250429160415835.png)



![image-20250429160524480](img/image-20250429160524480.png)

论文提到：RND obviates factors 2 and 3 since the target network can be chosen to be deterministic and inside  the model-class of the predictor network

也就是说，目标网络和预测网络需要是相同类型的深度神经网络，拟合的函数属于相同的函数族。这样避免不同的函数族之间的差距引入误差，保证用作内部奖励的误差主要是因为样本之前没有见过引入的。

**Model-class:** In machine learning, a "model-class" refers to the set of all possible functions that a particular type of model can represent.  For example, all possible neural networks with a specific architecture (e.g., a 3-layer fully connected network with 100 units per layer, using ReLU activations) belong to the same model-class.  By choosing a model-class, you're defining the family of functions that your learning algorithm can potentially learn.



![image-20250429165226737](img/image-20250429165226737.png)



算法伪代码：

![image-20250502133042249](img/image-20250502133042249.png)

### Experiments

不同的奖励组合方式下的表现：

1. 黄色（把内部奖励作为非回合的，并且使用双头价值输出）性能更好
2. 如果内外部奖励都作为回合性的奖励对待，单头的价值输出比双头的要稍微好一些

![image-20250429161933047](img/image-20250429161933047.png)

折扣率和环境并发收集数据对性能的影响：

1. 绿色高于蓝色：对内部奖励折扣越多，性能越好
2. 绿色高于橘色：对外部奖励折扣越少，性能越好

bison：我总觉得这个就简单的对比一下，没有说服力

![image-20250429162505222](img/image-20250429162505222.png)

与其他算法的对比，分别是橘色（PPO+RND探索补偿）、绿色（没有探索补偿的PPO）、蓝色（PPO+dynamics探索补偿）。6个有挑战的游戏中，4个RND都胜出。

![image-20250429163024323](img/image-20250429163024323.png)

### Discussion

RND方法适合局部探索，例如短期决策带来的后果，它不适合涉及到协同决策的全局探索。

另外，有个场景是4把钥匙6扇门，每开一扇门就会消耗一把钥匙。为了打开最后的两扇门，必须延迟满足，不要在当下就把四把钥匙都消耗了，尽管打开眼下的门可以立即获得奖励。为了鼓励agent延迟满足，必须对保存钥匙（而不是消耗钥匙）做足够多的内部奖励。这是未来重要的一个研究方向。
**DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**

### Introduction

三个主要的工作：

1. 只使用RL，不使用SFT，基于DeepSeek-V3-Base微调出DeepSeek-R1-zero，推理性能超过SOTA，但是出现了一些问题：可读性差、不同语言混合
2. 还是基于DeepSeek-V3-Base，使用少量标注数据进行SFT，外加RL，得到DeepSeek-R1，性能可以对标OpenAI-o1-1217
3. 基于Qwen2.5-32B这样的小模型作为student模型，蒸馏DeepSeek-R1，比对Qwen2.5-32B进行RL得到的性能更好。蒸馏的14B模型的表现比QwQ-32B-Preview还要好
4. 我们开源了上述模型



主要贡献：

1. 将大规模强化学习应用于基础大模型的后训练（Post-Training，对应前期的预训练）。第一次证实了只需要RL不需要SFT也可以获得很好的推理能力
2. 证实了通过蒸馏，小模型也可以很强大。通过从大模型蒸馏小模型，比对小模型进行RL，得到的性能更好。



评估结果：

1. 推理任务：
   1. DeepSeek-R1 achieves a score of 79.8% Pass@1 on AIME 2024, slightly surpassing OpenAI-o1-1217. On MATH-500, it attains an impressive score of 97.3%, performing on par with OpenAI-o1-1217 and significantly outperforming other models.
   2. On coding-related tasks, DeepSeek-R1 demonstrates expert level in code competition tasks,as it achieves 2,029 Elo rating on Codeforces outperforming 96.3% human participants in the competition
   3. For engineering-related tasks, DeepSeek-R1 performs slightly better than DeepSeek-V3, which could help developers in real world tasks.
2. 知识任务：
   1. DeepSeek-R1 achieves outstanding results, significantly outperforming DeepSeek-V3 with scores of 90.8% on MMLU, 84.0% on MMLU-Pro, and 71.5% on GPQA Diamond.While its performance is slightly below that of OpenAI-o1-1217 on these benchmarks, DeepSeek-R1surpasses other closed-source models, demonstrating its competitive edge in educational tasks.
   2. On the factual benchmark SimpleQA, DeepSeek-R1 outperforms DeepSeek-V3
3. 其他领域：
   1. DeepSeek-R1 achieves an impressive length-controlled win-rate of 87.6% on AlpacaEval 2.0 and a win-rate of 92.3% on Are-naHard
   2. DeepSeek-R1 demonstrates outstanding performance on tasks requiring long-context understanding, substantially outperforming DeepSeek-V3 on long-context benchmarks.

我的困惑：

Q：R1是基于DeepSeek-V3做后训练得到的，DeepSeek-R1在很多方面超过了DeepSeek-V3，这不是肯定和理所应当的？

AI：虽然 DeepSeek-R1 是基于 DeepSeek-V3 微调得到的，但其显著超越 V3 的推理能力，并不是“理所当然”的，而是大量设计和 RL 调优工作的成果。论文强调这一点，是为了凸显其 **推理驱动的后训练策略** 的成功。

Q：DeepSeek-V3-Base和DeepSeek-V3是什么关系和区别？

AI：DeepSeek-V3-Base 是预训练阶段的基础模型，而 DeepSeek-V3 是在此基础上经过监督微调（SFT）后的通用对齐模型。

![image-20250514084916435](img/image-20250514084916435.png)



### Approach
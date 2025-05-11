**Direct Preference Optimization:Your Language Model is Secretly a Reward Model**

### Introduction

预训练模型仅学习统计规律，缺乏对人类偏好（如安全、有用、无害）的显式对齐。预训练模型通过海量语料学习的本质是**"下一个token预测"的统计建模**，它捕捉的是文本表面的共现规律（如"巴黎-法国"的关联），而非人类期望的**价值对齐**（如事实准确、无害、有用）：

1. 预训练目标（极大似然估计）与真实需求（生成安全/有用的回答）存在本质鸿沟。例如模型可能为追求流畅性编造事实（"幻觉"），或生成政治不正确的文本。
2. 训练语料中的偏见和错误（如网络谣言）会被统计模型固化，而人类反馈可主动修正这些隐性缺陷。
3. 统计模型无法理解用户的偏好

RLHF/DPO通过人类反馈直接优化模型行为，使其更符合实际需求，避免生成低质或有害内容，提升实用性和安全性。

预训练模型的训练过程属于自监督学习，因为正确答案（下一个token）来自数据本身，而非外部标注。这就像人类通过阅读自学语言，而非依赖老师批改作业。后续的SFT或RLHF才引入有监督学习（人类标注数据）进行对齐。这种两阶段设计是大语言模型成功的核心：先通过无监督获得“语言能力”，再通过有监督/强化学习获得“人类偏好”。



DPO与RLHF不一样，采取监督学习里的二分类方法，让LLM遵从人类的偏好和控制，从而变得安全、有用、真实。

![image-20250511090213306](img/image-20250511090213306.png)

老看到这两个概念：

![image-20250511094202586](img/image-20250511094202586.png)

### Related works

主要提到了RLHF、RLAIF、CDB（contextual dueling bandi）、PbRL（preference-based RL）

```shell
传统RLHF流程：
专家编写demostration->SFT->人工标注偏好 → 训练奖励模型 → RL微调（PPO）
↑                          ↑
依赖大量人工                 依赖大量人工

LLM powerd FT：
人工提供文本规则 → LLM生成合成偏好 → (后续仍需要RL)
↑
弱监督

DPO的突破：
人工标注偏好 → 直接优化策略（跳过奖励模型和RL）
```

![image-20250511095456727](img/image-20250511095456727.png)

### Preliminaries（技术铺垫）

介绍了RLHF的三个阶段和对应的目标函数。

### Direct Preference Optimization

损失函数的数学推导：

![image-20250511122909640](img/image-20250511122909640.png)

强烈推荐[这个老师的教学视频](https://www.bilibili.com/video/BV1GF4m1L7Nt/?spm_id_from=333.337.search-card.all.click&vd_source=2173cb93b451f2278a1c87becf3ef529)

![image-20250511135042490](img/image-20250511135042490.png)

### Theoretical Analysis of DPO

与RLHF的数学等价性的证明，我看不太懂。略过

### Experiments

太累了，我发现即使是斯坦福大学这么牛逼的作者，也不太能把实验过程用图的方式一目了然的说清楚。这在商业化的职场是要被K的。

![image-20250511153129756](img/image-20250511153129756.png)
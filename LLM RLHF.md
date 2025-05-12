**Training language models to follow instructions with human feedback**

### Introduction

通过结合监督学习（人类演示数据）和强化学习（人类偏好排序数据）的微调方法（RLHF）, 可以使得大模型：

1. 更好的遵循人类的意图，对人类更有实质性的帮助
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

### bison的实验

用一个toy problem 来体验RLHF，我的奖励模型很简单：大模型输出的文本里如果有数字，就为正奖励，否则为负奖励。鼓励大模型用数据说话。



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from datetime import datetime
import re
import os
from transformers.modeling_outputs import CausalLMOutput # 这个可能不需要直接使用

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 100
EPOCHS = 3
CLIP_EPS = 0.2
KL_COEFF = 0.1 # KL 散度系数，可能需要调整
VF_COEFF = 0.5 # Value loss 系数，通常需要平衡 policy 和 value loss
SAVE_PATH = "./saved_model"
GAMMA = 0.99  # Discount factor (目前未在简单 reward 中使用，但 GAE 可能需要)
LR_ACTOR = 1e-5 # 学习率可以分开设置
LR_CRITIC = 1e-5

# ==== Dataset ====
class MyDataset(Dataset):
    def __init__(self, tokenizer, max_prompt_len=30, num_samples=500):
        self.tokenizer = tokenizer
        # 使用 try-except 避免因网络问题中断
        try:
            streamed_dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True) # 添加 trust_remote_code=True
            raw_dataset = [item for _, item in zip(range(num_samples), streamed_dataset)]
        except Exception as e:
            print(f"Error loading dataset: {e}. Using dummy data.")
            raw_dataset = [{"text": f"This is sample text number {i} for testing purposes."} for i in range(num_samples)]


        prompts = []
        for item in raw_dataset:
            # 简单的文本清理，去除多余空格
            clean_text = " ".join(item["text"].split())
            words = clean_text.split()
            if len(words) >= max_prompt_len:
                prompt = " ".join(words[:max_prompt_len])
                prompts.append(prompt)

        # 过滤掉可能为空的 prompt
        prompts = [p for p in prompts if p.strip()]
        if not prompts:
             raise ValueError("No valid prompts generated from the dataset.")

        encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len) # 明确 max_length
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

# ==== Reward Function ====
def compute_reward(texts):
    rewards = []
    for t in texts:
        # 查找独立的数字（前后是单词边界）
        matches = re.findall(r"\b\d+\b", t)
        count = len(matches)
        # 稍微调整奖励函数：鼓励至少有一个数字，多个数字奖励更高
        if count > 0:
            # 奖励随数字数量增加，但上限为 1.0
            reward = min(count * 0.2, 1.0)
        else:
            # 没有数字给予负奖励
            reward = -0.2
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

# ==== Actor Model ====
class ActorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2LMHeadModel(config)

    # 不需要重写 forward 和 generate，直接用 self.transformer 即可

# ==== Critic Model ====
class CriticModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config) # 使用 GPT2Model 获取 hidden states
        self.value_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用关键字参数调用
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_states = output.last_hidden_state
        # 通常取序列第一个或最后一个 token 的 hidden state 来预测整个序列的价值
        # 这里我们取最后一个 token (非 padding token) 的 hidden state
        # 注意：对于 left-padding，最后一个有效 token 不一定是 sequence_length - 1
        if attention_mask is None:
            values = self.value_head(hidden_states[:, -1, :]).squeeze(-1) # 取最后一个 hidden state
        else:
            # 找到每个序列的最后一个非 padding token 的索引
            sequence_lengths = attention_mask.sum(dim=1) - 1 # 索引从0开始
            # 使用 gather 获取对应位置的 hidden state
            last_token_hidden_states = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]
            values = self.value_head(last_token_hidden_states).squeeze(-1) # (batch_size,)
        return values


# ==== Function to get log probabilities of a sequence ====
def get_sequence_log_probs(model, input_ids, attention_mask):
    """计算给定模型下，序列 input_ids 的对数概率"""
    # 使用关键字参数调用 forward
    outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits # (batch_size, seq_len, vocab_size)
    # 计算除最后一个 token 外的所有 token 的 log prob
    # 因为 logits[i] 预测的是 token[i+1]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1) # (batch_size, seq_len-1, vocab_size)
    # 获取实际生成的 token (input_ids 的后半部分) 的 log prob
    # 注意：目标 token 是 input_ids[:, 1:]
    action_ids = input_ids[:, 1:].unsqueeze(-1) # (batch_size, seq_len-1, 1)
    # 使用 gather 获取对应 action 的 log prob
    gathered_log_probs = log_probs.gather(dim=-1, index=action_ids).squeeze(-1) # (batch_size, seq_len-1)

    # 应用 attention mask，只计算非 padding token 的 log prob
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float() # (batch_size, seq_len-1)
        gathered_log_probs = gathered_log_probs * mask

    # 对每个序列的 log prob 求和
    sequence_log_probs = gathered_log_probs.sum(dim=1) # (batch_size,)
    return sequence_log_probs

# ==== Training ====
def train():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # 对于 generate 需要 left-padding

    config = GPT2Config.from_pretrained("gpt2")
    config.pad_token_id = tokenizer.eos_token_id # 设置配置中的 pad_token_id

    # 初始化 Actor 和 Critic
    actor = ActorModel(config).to(DEVICE)
    critic = CriticModel(config).to(DEVICE)

    # 初始化旧 Actor (用于 PPO ratio 计算)
    old_actor = ActorModel(config).to(DEVICE)
    old_actor.load_state_dict(actor.state_dict()) # 初始时参数一致
    old_actor.eval() # 设置为评估模式

    # 设置优化器
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    writer = SummaryWriter(f"logs/ppo_separate_hf_adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print("Loading dataset...")
    dataset = MyDataset(tokenizer, max_prompt_len=20, num_samples=500) # 增加样本量可能更好
    # 数据集划分 (如果需要验证集)
    # train_size = int(0.9 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_set, _ = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # 使用完整数据集训练
    print("Dataset loaded.")

    global_step = 0
    for epoch in range(EPOCHS):
        print(f"Starting Epoch {epoch+1}/{EPOCHS}")
        # 在每个 epoch 开始时，同步 old_actor 的参数
        old_actor.load_state_dict(actor.state_dict())
        old_actor.eval()

        for i, batch in enumerate(train_loader):
            global_step += 1
            prompt_ids = batch["input_ids"].to(DEVICE)
            prompt_attention_mask = batch["attention_mask"].to(DEVICE)
            prompt_len = prompt_ids.shape[1]

            # --- 1. Rollout Phase: Generate text using the *current* actor ---
            actor.eval() # 切换到评估模式以关闭 dropout 等
            with torch.no_grad():
                # 使用 actor.transformer.generate
                generated_output = actor.transformer.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.0, # 可以调整温度
                    pad_token_id=tokenizer.pad_token_id, # 使用配置中的 pad_token_id
                    eos_token_id=tokenizer.eos_token_id # 明确结束符
                    # return_dict_in_generate=True, # 可选，如果需要更多生成信息
                    # output_scores=True # 可选
                )
                # generated_ids 包含 prompt + generated text
                generated_ids = generated_output # shape: (batch_size, prompt_len + gen_len)
                # 构建对应的 attention_mask
                # 新生成的 token mask 为 1，直到遇到 pad token
                generated_attention_mask = (generated_ids != tokenizer.pad_token_id).long()

                # 分离 prompt 和 generated part for log prob calculation later
                # response_ids = generated_ids[:, prompt_len:] # 只取生成的部分

                # 解码生成的完整文本 (prompt + response) 用于计算 reward
                full_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # 解码仅生成的部分文本 (response) 用于 tensorboard logging
                response_texts = tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)


            actor.train() # 切回训练模式

            # --- 2. Compute Rewards ---
            rewards = compute_reward(full_texts) # (batch_size,)

            # --- 3. Compute Log Probabilities and Values ---
            with torch.no_grad(): # 不需要计算这部分的梯度
                # 计算旧策略下生成序列的 log prob
                log_probs_old = get_sequence_log_probs(old_actor, generated_ids, generated_attention_mask)

                # 计算 Critic 对生成序列状态的价值估计
                # 输入完整的 generated_ids 和 mask 给 Critic
                values = critic(input_ids=generated_ids, attention_mask=generated_attention_mask) # (batch_size,)


            # --- 4. Compute Advantages ---
            # 这里使用简单的 TD(0) 优势，没有使用 GAE
            advantages = rewards - values # (batch_size,)
            # 标准化优势 (非常重要)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- 5. PPO Optimization Phase ---
            # 计算新策略下生成序列的 log prob (需要梯度)
            log_probs_new = get_sequence_log_probs(actor, generated_ids, generated_attention_mask)

            # 计算 Critic 的价值估计 (需要梯度，用于 Critic loss)
            values_new = critic(input_ids=generated_ids, attention_mask=generated_attention_mask)

            # --- Policy (Actor) Loss ---
            ratio = torch.exp(log_probs_new - log_probs_old.detach()) # (batch_size,)
            surrogate1 = ratio * advantages.detach() # 优势不需要梯度回传给 Actor
            surrogate2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages.detach()
            policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

            # --- Value (Critic) Loss ---
            # Critic 目标是拟合实际获得的 reward
            value_loss = F.mse_loss(values_new, rewards)

            # --- KL Divergence Penalty (Optional but Recommended) ---
            # 计算 KL 散度作为惩罚项 (衡量新旧策略的距离)
            # 可以近似为 log_probs_old - log_probs_new 的均值
            kl_div = torch.mean(log_probs_old.detach() - log_probs_new) # KL(old||new) 的近似
            # 或者严格计算 KL(new||old) = E[log P_new(a|s) - log P_old(a|s)]
            # kl_div = torch.mean(log_probs_new - log_probs_old.detach()) # 常用这个近似

            # --- Total Actor Loss ---
            # 注意：KL散度项是惩罚，应该加到损失上，但前面 policy_loss 是负的，所以这里是减去
            # 或者 policy_loss 取正，然后加上 KL 项
            # policy_loss = torch.mean(torch.min(surrogate1, surrogate2)) # 取正
            # actor_loss = policy_loss + KL_COEFF * kl_div # 如果 policy_loss 取正
            actor_loss = policy_loss + KL_COEFF * kl_div # 如果 policy_loss 是负的，KL 也应该与其符号一致或调整 KL 计算方式
                                                       # 标准 PPO paper 是 maximize objective - kl_penalty
                                                       # minimize -(objective - kl_penalty) = -objective + kl_penalty
                                                       # 所以这里应该是 policy_loss (负的) + KL_COEFF * kl_div

            # --- Backpropagation ---
            # Actor Update
            optimizer_actor.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0) # 可选：梯度裁剪
            optimizer_actor.step()

            # Critic Update
            optimizer_critic.zero_grad()
            critic_loss = VF_COEFF * value_loss # 使用系数调整 value loss 的影响
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0) # 可选：梯度裁剪
            optimizer_critic.step()


            # --- Logging ---
            if global_step % 20 == 0: # 减少日志频率
                print(f"Step: {global_step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, KL Div: {kl_div.item():.4f}, Mean Reward: {rewards.mean().item():.4f}")
                writer.add_scalar("reward/mean", rewards.mean().item(), global_step)
                writer.add_scalar("loss/policy", policy_loss.item(), global_step)
                writer.add_scalar("loss/value", value_loss.item(), global_step) # Log raw value loss
                writer.add_scalar("loss/actor_total", actor_loss.item(), global_step)
                writer.add_scalar("loss/critic_total", critic_loss.item(), global_step)
                writer.add_scalar("misc/kl_div", kl_div.item(), global_step)
                writer.add_scalar("misc/advantages_mean", advantages.mean().item(), global_step)
                # 记录一个样本
                writer.add_text("sample/prompt", tokenizer.decode(prompt_ids[0], skip_special_tokens=True), global_step)
                writer.add_text("sample/response", response_texts[0], global_step)
                writer.add_text("sample/full_text", full_texts[0], global_step)

        # --- Save models after each epoch ---
        epoch_save_path_actor = os.path.join(SAVE_PATH, "actor", f"epoch_{epoch+1}")
        epoch_save_path_critic = os.path.join(SAVE_PATH, "critic", f"epoch_{epoch+1}")
        os.makedirs(epoch_save_path_actor, exist_ok=True)
        os.makedirs(epoch_save_path_critic, exist_ok=True)

        print(f"Saving models for epoch {epoch+1}...")
        # 保存模型和 tokenizer
        actor.transformer.save_pretrained(epoch_save_path_actor)
        critic.transformer.save_pretrained(os.path.join(epoch_save_path_critic, "transformer")) # 保存 Critic 的 transformer 部分
        torch.save(critic.value_head.state_dict(), os.path.join(epoch_save_path_critic, "value_head.pth")) # 保存 value head
        tokenizer.save_pretrained(epoch_save_path_actor) # Tokenizer 和 Actor 通常一起保存
        tokenizer.save_pretrained(epoch_save_path_critic)
        print("Models saved.")


    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    train()
```

